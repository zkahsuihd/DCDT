
import os
import time
from module import *
import numpy as np
from test import *
from collections import defaultdict
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

###########################################################################################

def adjust_temperature(epoch, max_epochs, initial_temp, final_temp):
    """Cosine annealing for temperature parameter adjustment"""
    return final_temp + 0.5 * (initial_temp - final_temp) * \
           (1 + math.cos(math.pi * epoch / max_epochs))


def trainDCDT(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    # data of source subjects, which is used as the training set
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = DCDTPreTrainingModel(
        cuda,
        number_of_source=len(source_loader),
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        max_epochs=args.epoch_preTraining,
        lambda_disp=args.lambda_disp_pretrain,  # inputting

        temperature = args.temperature_pretrain  # Addition of a temperature parameter
    )
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    lambda_disp = 0.1  # dispersive loss weight
    for epoch in range(args.epoch_preTraining):
        print("epoch: " + str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        preTrainModel.set_current_epoch(epoch)  # Set current epoch for dynamic gradient revasal
        # Dynamically adjust temperature（optical）
        current_temp = adjust_temperature(
            epoch, args.epoch_preTraining,
            initial_temp=args.temperature_pretrain,
            final_temp=args.temperature_pretrain * 0.25
        )
        preTrainModel.diffusion.disp_loss.temperature = current_temp

        data_set_all = 0
        for i in range(1, iteration + 1):

            batch_dict = defaultdict(
                list)  # Pre-fetch a batch of data for each subject in advance and store them in this dictionary.
            data_dict = defaultdict(list)  # Store the data of each subject in the current batch
            label_dict = defaultdict(
                list)  # Store the labels corresponding to the data of each subject in the current batch
            label_data_dict = defaultdict(set)
            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1

            for j in range(len(source_iters)):
                # Assign a unique ID to each source subject
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()
                # the input of the model
                source_data, source_label = batch_dict[j]
                # Prepare corresponding new batch of each subject, the new batch has same label with current batch.
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                # Store the corresponding new batch of each subject, providing the supervision for different decoders
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data = corres_batch_data.cuda()
                data_set_all += len(source_label)
                optimizer_PreTraining.zero_grad()
                # Call the pretraining model

                rec_loss, sim_loss, disp_loss = preTrainModel(source_data, corres_batch_data, subject_id)

                # adding disp_loss
                loss_pretrain = rec_loss + args.beta * sim_loss + disp_loss  # disp_loss includes lambda_disp
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: " + str(data_set_all))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DCDT/loss',
                           {'loss_pretrain': loss_pretrain.data,
                            'rec_loss': rec_loss.data,
                            'sim_loss': sim_loss.data,
                            'disp_loss': disp_loss.data,
                            'temperature': current_temp  },
                           epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: " + str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    # Load the ABP module, the encoder from pretrained model and build a new model for the fine-tuning phase
    fineTuneModel = DCDTFineTuningModel(
        cuda,
        preTrainModel,
        number_of_category=args.cls_classes,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        max_epochs=args.epoch_fineTuning,
        lambda_disp=args.lambda_disp_finetune,
        temperature = args.temperature_finetune
    )

    #  Keep temperature constant or slightly adjust during fine-tuning

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        fineTuneModel.set_current_epoch(epoch)  # Set current epoch for dynamic gradient revasal

        count = 0
        data_set_all = 0
        current_temp = adjust_temperature(
            epoch, args.epoch_preTraining,
            initial_temp=args.temperature_finetune,
            final_temp=args.temperature_finetune * 0.25
        )
        fineTuneModel.diffusion.disp_loss.temperature = current_temp
        current_temp = args.temperature_finetune
        for i in range(1, iteration + 1):
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                # Assign a unique ID to each source subject
                subject_id = torch.ones(args.batch_size) * j
                subject_id = subject_id.long()
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                # Call the fine-tuning model
                x_pred, x_logits, losses = fineTuneModel(
                    source_data,
                    label_src=source_label,
                    subject_id=subject_id,
                    lambda_domain=args.lambda_domain
                )
                # Calculate total loss (classification loss + domain adversarial loss)
                total_loss = losses['cls']
                if 'domain' in losses:
                    # Updated loss calculation (incorporating disp_loss)
                    total_loss = losses['cls'] + \
                                 losses.get('domain', 0) * args.lambda_domain + \
                                 losses.get('disp', 0)
                total_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        # Logging All Loss Terms
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DCDT/loss',
                           {'cls_loss': losses['cls'].data,
                            'total_loss': total_loss.data,
                            'temperature': current_temp  },
                           epoch + 1)
        if 'domain' in losses:
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DCDT/domain_loss',
                              losses['domain'].data, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DCDT/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))
        # test the fine-tuned model with the data of unseen target subject
        testModel = DCDTTestModel(fineTuneModel)
        acc_DCDT = testDCDT(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DCDT: " + str(acc_DCDT))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DCDT/test acc',
                           {'test acc': acc_DCDT}, epoch + 1)
        if acc_DCDT > acc_final:
            acc_final = acc_DCDT
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    # save models
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final


##################################################################################################
