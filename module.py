import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import copy


class DispersiveLoss(nn.Module):
    def __init__(self, lambda_disp=0.5, temperature=0.1, reduction='mean'):
        """
        Args:
            lambda_disp: Regularization strength (default 0.5, consistent with optimal value in the paper)
            temperature: Temperature parameter τ (default 0.1)
            reduction: Reduction method for loss ('mean' or 'sum')
        """
        super().__init__()
        self.lambda_disp = lambda_disp
        self.temperature = temperature
        self.reduction = reduction
        self.eps = 1e-8  # Numerical stability term

    def forward(self, latent_representations):
        """
        Args:
            latent_representations: [batch_size, seq_len, latent_dim]
        Returns:
            InfoNCE-based dispersive loss (with temperature parameter τ)
        """
        batch_size, seq_len, _ = latent_representations.shape

        # Normalize latent vectors
        latent_norm = F.normalize(latent_representations, p=2, dim=-1)

        # Compute similarity matrix (apply temperature parameter τ)
        similarity = torch.bmm(latent_norm, latent_norm.transpose(1, 2)) / self.temperature

        # Create mask to exclude diagonal (self-similarity)
        mask = torch.eye(seq_len, device=latent_representations.device).bool()
        similarity = similarity.masked_fill(mask.unsqueeze(0), -float('inf'))

        # Compute InfoNCE loss
        loss = -F.log_softmax(similarity, dim=-1)

        # Aggregate loss according to reduction method
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum() / (batch_size * seq_len)

        return self.lambda_disp * loss
# Dynamic Gradient Reversal Layer (improved with cosine annealing strategy)
class DynamicGradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, current_epoch, max_epochs, min_alpha=0.1, max_alpha=1.0, mode='pretrain'):
        """
        Args:
            x: Input features
            current_epoch: Current training epoch
            max_epochs: Total number of training epochs
            min_alpha: Minimum reversal strength
            max_alpha: Maximum reversal strength
            mode: 'pretrain' or 'finetune', determines the direction of alpha change
        """
        # Cosine annealing strategy
        progress = current_epoch / max_epochs
        if mode == 'finetune':
            # Fine-tuning phase: alpha linearly decays from 0.5 to 0.1
            alpha = max_alpha - (max_alpha - min_alpha) * progress
        else:
            # Pre-training phase: alpha increases from 0.1 to 1.0 (original logic)
            alpha = min_alpha + 0.5 * (max_alpha - min_alpha) * (1 + math.cos(math.pi * progress))

        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Apply dynamic gradient reversal
        output = grad_output.neg() * ctx.alpha
        return output, None, None, None, None, None

# Timestep Embedding Module
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device)) * -embeddings
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ABP module
class Attention(nn.Module):
    def __init__(self, cuda, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        if cuda:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim).cuda())
            self.u_linear = nn.Parameter(torch.randn(input_dim).cuda())
        else:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim))
            self.u_linear = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, batch_size, time_steps):
        x_reshape = torch.Tensor.reshape(x, [-1, self.input_dim])
        attn_softmax = F.softmax(torch.mm(x_reshape, self.w_linear) + self.u_linear, 1)
        res = torch.mul(attn_softmax, x_reshape)
        res = torch.Tensor.reshape(res, [batch_size, time_steps, self.input_dim])
        return res


# CDT module
class CDTModel(nn.Module):
    def __init__(self, input_dim=310, latent_dim=128, time_dim=128, nhead=8, num_layers=3,
                 use_disp_loss=True, lambda_disp=0.5, temperature=0.1, disp_reduction='mean'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_disp_loss = use_disp_loss

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            # TanhExponentialLU(),
            nn.Linear(time_dim, latent_dim)
        )

        # Input encoder (for noisy input)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            # TanhExponentialLU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim)
        )

        # Condition projection layer (for conditional input)
        self.condition_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            # TanhExponentialLU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim)
        )

        # Transformer fusion module
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation='gelu',
            # activation=TanhExponentialLU(),
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(latent_dim, dropout=0.1)

        # Noise prediction head (decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            # TanhExponentialLU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, input_dim)
        )

        # Add dispersive loss module
        if use_disp_loss:
            self.disp_loss = DispersiveLoss(
                lambda_disp=lambda_disp,
                temperature=temperature,  # Temperature parameter
                reduction=disp_reduction   # Loss reduction method
            )

    def forward(self, x, t, condition=None):
        """
        Args:
            x: Noisy input [batch_size, seq_len, input_dim]
            t: Timestep [batch_size]
            condition: Conditional input [batch_size, seq_len, input_dim]
        Returns:
            Predicted noise (and dispersive loss if enabled and training)
        """
        # 1. Encode input features
        encoded_x = self.encoder(x)  # [batch_size, seq_len, latent_dim]

        # 2. Process time embeddings
        t_embed = self.time_mlp(t)  # [batch_size, latent_dim]
        t_embed = t_embed.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, seq_len, latent_dim]

        # 3. Process conditional input
        if condition is None:
            cond_embed = torch.zeros_like(encoded_x)
        else:
            cond_embed = self.condition_proj(condition)  # [batch_size, seq_len, latent_dim]

        # 4. Fuse features (input + time + condition)
        fused = encoded_x + t_embed + cond_embed  # [batch_size, seq_len, latent_dim]

        # 5. Transformer processing (add positional encoding)
        fused = fused.permute(1, 0, 2)  # [seq_len, batch_size, latent_dim] (Transformer input)
        fused = self.pos_encoder(fused)
        transformed = self.transformer(fused)  # [seq_len, batch_size, latent_dim]
        transformed = transformed.permute(1, 0, 2)  # [batch_size, seq_len, latent_dim]

        # 6. Decode to predict noise
        pred_noise = self.decoder(transformed)  # [batch_size, seq_len, input_dim]

        if self.use_disp_loss and self.training:
            disp_loss = self.disp_loss(transformed)
            return pred_noise, disp_loss

        return pred_noise

    # Modified encode method for TransformerDiffusionModel
    def encode(self, x, condition=None, return_disp_loss=False):
        """
        Extract features (using only encoder and condition projection).
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            condition: Conditional input [batch_size, seq_len, input_dim]
            return_disp_loss: If True, also return dispersive loss (when training)
        Returns:
            Fused features (and dispersive loss if requested)
        """
        # 1. Encode input features
        encoded_x = self.encoder(x)  # [batch_size, seq_len, latent_dim]

        # 2. Process conditional input
        if condition is None:
            cond_embed = torch.zeros_like(encoded_x)
        else:
            cond_embed = self.condition_proj(condition)

        # 3. Fuse features
        fused = encoded_x + cond_embed

        if return_disp_loss and self.training and self.use_disp_loss:
            disp_loss = self.disp_loss(fused)
            return fused, disp_loss
        elif return_disp_loss:
            # Return 0 if not training or disp_loss not enabled
            return fused, torch.tensor(0.0, device=fused.device)
        return fused


# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Domain Classifier
class DomainClassifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=14):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)


# emotion Classifier
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# Pre-training Model
class DCDTPreTrainingModel(nn.Module):
    def __init__(self, cuda, number_of_source=14, batch_size=512, time_steps=30, max_epochs=50, lambda_disp=0.5, temperature=0.1):
        super(DCDTPreTrainingModel, self).__init__()
        self.batch_size = batch_size
        self.number_of_source = number_of_source
        self.time_steps = time_steps
        self.max_epochs = max_epochs
        self.current_epoch = 0  # Track current epoch

        # Dispersive loss weight
        self.lambda_disp = lambda_disp

        # Retain original ABP module
        self.attentionLayer = Attention(cuda, input_dim=310)

        # Diffusion model as the core module
        self.CDT = CDTModel(
            input_dim=310,
            latent_dim=128,
            nhead=8,
            num_layers=1,
            use_disp_loss=True,      # Enable dispersive loss
            lambda_disp=self.lambda_disp,
            temperature=temperature, # Pass temperature parameter
            disp_reduction='mean'    # Use mean reduction explicitly
        )

        # Domain classifier
        self.domainClassifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, number_of_source)
        )

        # Loss function
        self.mse = nn.MSELoss()

        if cuda:
            self.cuda()

        # Define beta schedule
        betas = self._cosine_beta_schedule(time_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))

    def set_current_epoch(self, epoch):
        """Set the current epoch for dynamic adjustment of gradient reversal strength"""
        self.current_epoch = epoch

    def _cosine_beta_schedule(self, time_steps, s=0.008):
        """Generate beta values with cosine schedule"""
        steps = time_steps + 1
        x = torch.linspace(0, time_steps, steps)
        alphas_cumprod = torch.cos(((x / time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def add_noise(self, x, t):
        """Add noise to the input"""
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod.to(x.device)[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x.device)[t].view(-1, 1, 1)

        noise = torch.randn_like(x)
        noisy_x = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        return noisy_x, noise

    def forward(self, x, corres, subject_id):
        x_batch_size = x.size(0)
        corres_batch_size = corres.size(0)

        # Apply ABP module to input and corres
        x_attn = self.attentionLayer(x, x_batch_size, self.time_steps)
        corres_attn = self.attentionLayer(corres, corres_batch_size, self.time_steps)
        splitted_tensors = torch.chunk(corres_attn, self.number_of_source, dim=0)

        recon_loss = 0
        disp_loss_total = 0  # Track total dispersive loss

        # Add noise to data from each source domain
        for i in range(self.number_of_source):
            # Randomly sample timesteps
            t = torch.randint(0, self.time_steps, (x_batch_size,), device=x.device).long()
            target = splitted_tensors[i]
            noisy_corres, noise = self.add_noise(target, t)

            # Predict noise (using x_attn as condition)
            pred_noise, disp_loss = self.CDT(noisy_corres, t, condition=x_attn)
            # Reconstruction loss (predicted noise vs. actual noise)
            recon_loss += self.mse(pred_noise, noise)
            disp_loss_total += disp_loss  # Accumulate dispersive loss

        # Extract latent features (using conditional information)
        latent = self.CDT.encode(x_attn, condition=x_attn)
        latent = latent.mean(dim=1)  # Mean along time dimension

        # Domain adversarial training (using dynamic gradient reversal)
        reverse_feature = DynamicGradientReversalLayer.apply(
            latent,
            self.current_epoch,
            self.max_epochs,
            0.1,  # min_alpha
            1.0   # max_alpha
        )
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict, dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        return recon_loss, sim_loss, disp_loss_total  # Return total (summed) dispersive loss

# Fine-tuning Model
class DCDTFineTuningModel(nn.Module):
    def __init__(self, cuda, baseModel, number_of_category=3, batch_size=10, time_steps=15, max_epochs=50, lambda_disp=0.1, temperature=0.1):
        super(DCDTFineTuningModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)
        self.attentionLayer = self.baseModel.attentionLayer
        self.CDT = self.baseModel.CDT
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.max_epochs = max_epochs
        self.current_epoch = 0

        # Ensure diffusion model enables dispersive loss
        self.CDT.use_disp_loss = True
        # Set dispersive loss weight
        self.lambda_disp = lambda_disp
        self.CDT.disp_loss.lambda_disp = self.lambda_disp  # Set lambda_disp
        # Ensure diffusion model uses the same reduction method
        self.CDT.disp_loss.reduction = 'mean'  # Use mean reduction
        # Inherit temperature parameter from base model
        self.CDT = copy.deepcopy(baseModel.CDT)
        self.CDT.disp_loss.temperature = temperature  # Override temperature if needed

        # Emotion classifier
        self.emotionClassifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, number_of_category)
        )

        # Domain classifier (for adversarial training during fine-tuning)
        self.domainClassifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),  # 0.3
            nn.Linear(64, self.baseModel.number_of_source)
        )

        if cuda:
            self.cuda()

    def set_current_epoch(self, epoch):
        """Set current epoch for dynamic adjustment of gradient reversal strength"""
        self.current_epoch = epoch

    def forward(self, x, label_src=None, subject_id=None, lambda_domain=0.5):
        # 1. Apply ABP module
        x_attn = self.attentionLayer(x, x.size(0), self.time_steps)

        # 2. Extract features via diffusion model

        # 2. Extract features via diffusion model (handle disp_loss correctly)
        if self.training and self.CDT.use_disp_loss:
            latent, disp_loss = self.CDT.encode(
                x_attn,
                condition=x_attn,
                return_disp_loss=True
            )
        else:
            latent = self.CDT.encode(x_attn, condition=x_attn)
            disp_loss = torch.tensor(0.0, device=x.device)
        latent_mean = latent.mean(dim=1)  # Mean along time dimension

        # 3. Emotion classification
        logits = self.emotionClassifier(latent_mean)
        pred = F.log_softmax(logits, dim=1)

        losses = {}
        # Add dispersive loss to return dictionary
        losses['disp'] = disp_loss  # Apply weight as needed

        # 4. Compute classification loss
        if label_src is not None:
            cls_loss = F.cross_entropy(logits, label_src.squeeze())
            losses['cls'] = cls_loss

        # 5. Domain adversarial training (used only during fine-tuning)
        if subject_id is not None:
            reverse_feature = DynamicGradientReversalLayer.apply(
                latent_mean,
                self.current_epoch,
                self.max_epochs,
                0.1,  # min_alpha
                0.5,  # max_alpha
                'finetune'
            )
            domain_predict = self.domainClassifier(reverse_feature)
            domain_predict = F.log_softmax(domain_predict, dim=1)
            domain_loss = F.nll_loss(domain_predict, subject_id)
            losses['domain'] = domain_loss * lambda_domain  # Adjustable weight

        return pred, logits, losses


# Test Model
class DCDTTestModel(nn.Module):
    def __init__(self, baseModel):
        super(DCDTTestModel, self).__init__()
        self.baseModel = baseModel

    def forward(self, x):
        # Ensure evaluation mode
        self.baseModel.eval()
        with torch.no_grad():
            # 1. Apply ABP module
            x_attn = self.baseModel.attentionLayer(x, x.size(0), self.baseModel.time_steps)

            # 2. Extract features via diffusion model
            latent = self.baseModel.CDT.encode(x_attn)
            latent = latent.mean(dim=1)  # Temporal average pooling

            # 3. Emotion classification
            logits = self.baseModel.emotionClassifier(latent)
            return logits


# Feature Visualization Model
class ModelReturnFeatures(nn.Module):
    def __init__(self, baseModel, time_steps=15):
        super(ModelReturnFeatures, self).__init__()
        self.baseModel = baseModel
        self.time_steps = time_steps

        for param in self.baseModel.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_attn = self.baseModel.attentionLayer(x, x.size(0), self.time_steps)
        latent = self.baseModel.CDT.encode(x_attn)
        latent_mean = latent.mean(dim=1)
        return x_attn, latent_mean
