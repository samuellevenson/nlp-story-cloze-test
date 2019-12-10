import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from transformers import (WEIGHTS_NAME, BertConfig, BertForMultipleChoice, BertTokenizer)
from transformers import AdamW
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

from tqdm import tqdm, trange

from part_b_utils import (convert_examples_to_features, RocStoriesProcessor)

logger = logging.getLogger(__name__)

LOWER_CASE = True
NUM_LABELS = 2

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def train(args, train_dataset, model, tokenizer):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimzer and schedule (linear warrmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # update learning rate schedule
                model.zero_grad()
                global_step += 1

        results = evaluate(args, model, tokenizer)
        print("epoch acc: " + str(results["eval_acc"]))
        print("epoch loss: " + str(results["eval_loss"]))
            
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix="", test=False):
    results = {}
    eval_dataset = load_examples(args, tokenizer, evaluate=not test, test=test)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # evaluate
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    story_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
            story_ids = inputs['input_ids'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            story_ids = np.append(story_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    if evaluate:
        acc = simple_accuracy(preds, out_label_ids)
    if test:
        to_csv(preds, story_ids, "results.csv")
    result = {"eval_acc": acc, "eval_loss": eval_loss}
    results.update(result)

    output_eval_file = os.path.join(args.output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
        writer.write("model           =%s\n" % str(args.model_name_or_path))
        writer.write("total batch size=%d\n" % (args.batch_size * args.gradient_accumulation_steps))
        writer.write("train num epochs=%d\n" % args.num_train_epochs)
        writer.write("max seq length  =%d\n" % args.max_seq_length)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_examples(args, tokenizer, evaluate=False, test=False):

    processor = RocStoriesProcessor()
    # load data features dataset file

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    elif test:
        examples = processor.get_test_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)
    logger.info("Training number: %s", str(len(examples)))
    features = convert_examples_to_features(
        examples,
        label_list,
        args.max_seq_length,
        tokenizer,
        pad_on_left=False,
        pad_token_segment_id=0
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def to_csv(preds, ids, csv_name):
    csv_output = "preds\n"
    for i in range(0, len(preds) - 1):
        csv_output += str(preds[i]) + "\n"
    f = open(csv_name, "w")
    f.write(csv_output)
    f.close()

def main():
    # set up parser
    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    #optional args
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--from_save", action='store_true')
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    args = parser.parse_args()

    logger.info("Training/evaluation parameters %s", args)

    # load pretrained model and tokenizer
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_LABELS)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=LOWER_CASE)
    model = BertForMultipleChoice.from_pretrained(args.model_name_or_path, config=config)

    # training
    if not args.from_save:
        train_dataset = load_examples(args, tokenizer)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # save model
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        model = BertForMultipleChoice.from_pretrained(args.save_dir)
        tokenizer = BertTokenizer.from_pretrained(args.save_dir)

    # evaluation
    results = {}
    result = evaluate(args, model, tokenizer)
    result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    results.update(result)

    # testing
    if args.do_test:
        result = evaluate(args, model, tokenizer, test=True)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)
    
    return results

if __name__ == "__main__":
    main()