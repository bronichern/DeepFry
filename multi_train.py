import torch
import numpy as np
from utils import CREAKY
import utils as utils


def create_predictions(test_loader, model, criterion, device, tv=0.5, tc=0.5):
    with torch.no_grad():
        model.eval()

        global_epoch_loss = 0
        creaky_acc = 0
        total = 0
        tp = np.zeros(1)
        fp = np.zeros(1)
        fn = np.zeros(1)
        tn = np.zeros(1)
        predict_dict = {}

        for raw, creaky_labels, lens_list, filenames, starts in test_loader:
            raw, creaky_labels, = raw.to(device), creaky_labels.to(device)
            if raw.shape[-1] == 0:
                continue
            logits, output = model(raw)

            # if raw.shape[0] == 1:
            #     output = output.unsqueeze(0)
                # losits = losits.unsqueeze(0)
            for idx in range(output.size(0)):
                cur_len = lens_list[idx]
                current_creaky_label = creaky_labels[idx, :cur_len]
                current_output = output[idx, :cur_len]
                global_epoch_loss += criterion(
                    current_output[:, 1], current_creaky_label)

                creaky_pred = torch.logical_and(
                    current_output[:, 0] >= tv, current_output[:, 1] >= tc).int()
                voice_pred = (current_output[:, 0] >= tv).int()
                total += current_output[:, 0].numel()

                creaky_eq_tensors = (creaky_pred == current_creaky_label)
                creaky_acc += torch.sum(creaky_eq_tensors).item()
                creaky_idx = torch.nonzero(
                    (current_creaky_label == CREAKY).squeeze())

                tp[0] += torch.sum(creaky_eq_tensors[creaky_idx]).item()
                fp[0] += torch.logical_and((torch.logical_not(current_creaky_label) == creaky_pred),
                                           torch.logical_not(current_creaky_label)).float().sum().item()
                fn[0] += torch.logical_and((torch.logical_not(
                    creaky_pred) == current_creaky_label), current_creaky_label).float().sum().item()
                tn[0] += torch.logical_and(current_creaky_label == creaky_pred,
                                           torch.logical_not(current_creaky_label)).float().sum().item()

                file_predictions = predict_dict.get(filenames[idx], [])
                file_predictions.append([starts[idx], creaky_pred.cpu().numpy(), current_creaky_label.cpu().numpy()])
                predict_dict[filenames[idx]] = file_predictions

    global_test_loss = global_epoch_loss / len(test_loader)
    creaky_accuracy = creaky_acc/total

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return global_test_loss.cpu(), creaky_accuracy, total, precision, recall, f1, predict_dict


def test_textgrids(test_loader, model, criterion, device, window_size, output_path=None, tv=0.5, tc=0.5):
    global_test_loss, creaky_accuracy, total, precision, recall, f1, predict_dict = create_predictions(
        test_loader, model, criterion, device, tv, tc)

    word_precision, word_recall, phone_precision, phone_recall = utils.word_pho_acc(
        predict_dict, window_size)
    phone_f1 = 2 * (phone_precision * phone_recall) / (phone_precision +
                                                       phone_recall) if (phone_precision + phone_recall) > 0 else 0
    word_f1 = 2 * (word_precision * word_recall) / (word_precision +
                                                    word_recall) if (word_precision + word_recall) > 0 else 0

    total_creaky_pred = []
    total_creaky_label = []

    for filename, data_list in predict_dict.items():
        for start, creaky_pred, creaky_label in data_list:
            total_creaky_pred.extend(creaky_pred)
            total_creaky_label.extend(creaky_label)

    print( f'\nTest set: Average loss: {global_test_loss:.4f}, Creaky Accuracy: {creaky_accuracy}, Creaky f1: {f1[0]}\n ')
    print(f'\n Creaky Precision: {precision[0]}, Creaky Recall: {recall[0]}\n ')

    print(f"WORD - precision: {word_precision:.4f}, recall: {word_recall:.4f}, f1: {word_f1:.4f}")
    print(f"PHONES - precision: {phone_precision:.4f}, recall: {phone_recall:.4f}, f1: {phone_f1:.4f}")

    if output_path:
        print(f"Creating textgridts into {output_path}")
        utils.create_textgrid(predict_dict, output_path, window_size)

    return global_test_loss, creaky_accuracy, f1[0], phone_f1, word_f1
