import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0


class T5Dataset(Dataset):
    """
    Clean, correct dataset implementation for T5 text-to-SQL.
    """

    def __init__(self, data_folder, split):
        self.data_folder = data_folder
        self.split = split

        # T5 small tokenizer (SentencePiece)
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        # T5 uses <pad> (ID 0) as BOS and <eos> as </s> (ID 1)
        self.decoder_start_id = self.tokenizer.pad_token_id  # 0

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []

        self.process_data()

    def process_data(self):
        """Load NL and SQL, tokenize, and build encoder/decoder sequences."""
        nl_path = os.path.join(self.data_folder, f"{self.split}.nl")
        nl_lines = self._load_lines(nl_path)

        # ----------------------
        #  TEST SET: no SQL
        # ----------------------
        if self.split == "test":
            for nl in nl_lines:
                enc_text = "translate English to SQL: " + nl
                enc_ids = self.tokenizer(
                    enc_text,
                    add_special_tokens=True
                )["input_ids"]

                self.encoder_inputs.append(torch.tensor(enc_ids))
            return

        # ----------------------
        #  TRAIN/DEV SETS
        # ----------------------
        sql_path = os.path.join(self.data_folder, f"{self.split}.sql")
        sql_lines = self._load_lines(sql_path)

        for nl, sql in zip(nl_lines, sql_lines):
            # Prefix prompt (recommended for T5)
            enc_text = "translate English to SQL: " + nl

            # ----- Tokenize NL (encoder input) -----
            enc_ids = self.tokenizer(
                enc_text,
                add_special_tokens=True
            )["input_ids"]

            # ----- Tokenize SQL (target) -----
            # add_special_tokens=True ensures an </s> EOS token is included
            tgt_ids = self.tokenizer(
                sql,
                add_special_tokens=True
            )["input_ids"]

            # ----- SHIFT RIGHT decoder input -----
            dec_in_ids = [self.decoder_start_id] + tgt_ids[:-1]

            # Save tensors
            self.encoder_inputs.append(torch.tensor(enc_ids))
            self.decoder_inputs.append(torch.tensor(dec_in_ids))
            self.decoder_targets.append(torch.tensor(tgt_ids))

    def _load_lines(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split == "test":
            enc = self.encoder_inputs[idx]
            bos = torch.tensor([self.decoder_start_id])
            return enc, bos

        enc = self.encoder_inputs[idx]
        dec_in = self.decoder_inputs[idx]
        dec_tg = self.decoder_targets[idx]
        bos = torch.tensor([self.decoder_start_id])

        return enc, dec_in, dec_tg, bos


# ======================================================
#  COLLATE FUNCTIONS
# ======================================================

def normal_collate_fn(batch):
    """
    Collate function for train/dev.
    Returns:
        encoder_ids (B, T)
        encoder_mask (B, T)
        decoder_inputs (B, T')
        decoder_targets (B, T')
        bos_token_for_eval (B, 1)
    """
    encs, dec_ins, dec_tgs, bos = zip(*batch)

    encs = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    enc_mask = (encs != PAD_IDX).long()

    dec_ins = pad_sequence(dec_ins, batch_first=True, padding_value=PAD_IDX)
    dec_tgs = pad_sequence(dec_tgs, batch_first=True, padding_value=PAD_IDX)

    bos = torch.stack(bos)

    return encs, enc_mask, dec_ins, dec_tgs, bos


def test_collate_fn(batch):
    """
    Collate function for test set (no SQL decoder inputs).
    """
    encs, bos_tokens = zip(*batch)

    encs = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    enc_mask = (encs != PAD_IDX).long()

    bos_tokens = torch.stack(bos_tokens)

    return encs, enc_mask, bos_tokens


# ======================================================
#  DATALOADER WRAPPERS
# ======================================================

def get_dataloader(batch_size, split):
    dset = T5Dataset("data", split)
    shuffle = split == "train"
    collate = normal_collate_fn if split != "test" else test_collate_fn

    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate
    )


def load_t5_data(batch_size, test_batch_size):
    return (
        get_dataloader(batch_size, "train"),
        get_dataloader(test_batch_size, "dev"),
        get_dataloader(test_batch_size, "test"),
    )



# import os, random
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from transformers import T5TokenizerFast
# import torch

# PAD_IDX = 0


# class T5Dataset(Dataset):
#     """
#     Dataset for T5 text-to-SQL training and inference.
#     Returns:
#         (train/dev):
#             encoder_ids
#             encoder_mask
#             decoder_inputs
#             decoder_targets
#             initial_decoder_inputs
#         (test):
#             encoder_ids
#             encoder_mask
#             initial_decoder_inputs
#     """

#     def __init__(self, data_folder, split):
#         self.data_folder = data_folder
#         self.split = split
#         self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

#         # this is T5's "start" token (we choose extra_id_0)
#         self.decoder_start_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

#         self.encoder_inputs = []
#         self.decoder_inputs = []
#         self.decoder_targets = []

#         self.process_data()


#     def process_data(self):
#         """Load .nl and .sql files and tokenize."""
#         nl_path = os.path.join(self.data_folder, f"{self.split}.nl")

#         self.nl_lines = self._load_lines(nl_path)

#         # test set has no SQL file
#         if self.split == "test":
#             for nl in self.nl_lines:
#                 encoder_text = "translate English to SQL: " + nl
#                 enc = self.tokenizer(encoder_text, return_tensors=None)["input_ids"]
#                 self.encoder_inputs.append(torch.tensor(enc))
#             return

#         # train/dev sets
#         sql_path = os.path.join(self.data_folder, f"{self.split}.sql")
#         sql_lines = self._load_lines(sql_path)

#         for nl, sql in zip(self.nl_lines, sql_lines):

#             # ----- Encoder input -----
#             encoder_text = "translate English to SQL: " + nl
#             encoder_ids = self.tokenizer(encoder_text, return_tensors=None)["input_ids"]

#             # ----- Decoder targets -----
#             target_ids = self.tokenizer(sql, return_tensors=None)["input_ids"]

#             # ----- Decoder inputs: shift right -----
#             decoder_input = [self.decoder_start_id] + target_ids[:-1]

#             self.encoder_inputs.append(torch.tensor(encoder_ids))
#             self.decoder_inputs.append(torch.tensor(decoder_input))
#             self.decoder_targets.append(torch.tensor(target_ids))


#     def _load_lines(self, path):
#         with open(path, "r") as f:
#             return [line.strip() for line in f.readlines()]


#     def __len__(self):
#         return len(self.nl_lines)


#     def __getitem__(self, idx):
#         if self.split == "test":
#             enc = self.encoder_inputs[idx]
#             return enc, None, None, None, torch.tensor([self.decoder_start_id])

#         enc = self.encoder_inputs[idx]
#         dec_in = self.decoder_inputs[idx]
#         dec_tg = self.decoder_targets[idx]
#         initial_dec = torch.tensor([self.decoder_start_id])
#         return enc, dec_in, dec_tg, initial_dec


# def normal_collate_fn(batch):
#     """
#     For train & dev sets.
#     Returns:
#         encoder_ids, encoder_mask,
#         decoder_inputs, decoder_targets,
#         initial_decoder_inputs
#     """
#     encs, dec_ins, dec_tgs, init_decs = zip(*batch)

#     # Pad encoder
#     encs = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
#     enc_mask = (encs != PAD_IDX).long()

#     # Pad decoder
#     dec_ins = pad_sequence(dec_ins, batch_first=True, padding_value=PAD_IDX)
#     dec_tgs = pad_sequence(dec_tgs, batch_first=True, padding_value=PAD_IDX)

#     init_decs = torch.stack(init_decs)

#     return encs, enc_mask, dec_ins, dec_tgs, init_decs


# def test_collate_fn(batch):
#     """
#     For test set only.
#     Returns:
#         encoder_ids, encoder_mask, initial_decoder_inputs
#     """
#     encs = []
#     init_decs = []

#     for enc, _, _, _, init_dec in batch:
#         encs.append(enc)
#         init_decs.append(init_dec)

#     encs = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
#     enc_mask = (encs != PAD_IDX).long()
#     init_decs = torch.stack(init_decs)

#     return encs, enc_mask, init_decs


# def get_dataloader(batch_size, split):
#     data_folder = "data"
#     dataset = T5Dataset(data_folder, split)
#     shuffle = split == "train"
#     collate = normal_collate_fn if split != "test" else test_collate_fn

#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# def load_t5_data(batch_size, test_batch_size):
#     train_loader = get_dataloader(batch_size, "train")
#     dev_loader = get_dataloader(test_batch_size, "dev")
#     test_loader = get_dataloader(test_batch_size, "test")
#     return train_loader, dev_loader, test_loader


# import os, random, re, string
# from collections import Counter
# from tqdm import tqdm
# import pickle

# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence

# import nltk
# nltk.download('punkt')
# from transformers import T5TokenizerFast
# import torch

# PAD_IDX = 0

# class T5Dataset(Dataset):

#     def __init__(self, data_folder, split):
#         '''
#         Skeleton for the class for performing data processing for the T5 model.

#         Some tips for implementation:
#             * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
#               the encoder and decoder output. 
#             * You want to provide the decoder some beginning of sentence token. Any extra-id on the
#               T5Tokenizer should serve that purpose.
#             * Class behavior should be different on the test set.
#         '''
#         # TODO

#     def process_data(self, data_folder, split, tokenizer):
#         # TODO
    
#     def __len__(self):
#         # TODO

#     def __getitem__(self, idx):
#         # TODO

# def normal_collate_fn(batch):
#     '''
#     Collation function to perform dynamic padding for training and evaluation with the
#     development or validation set.

#     Inputs:
#         * batch (List[Any]): batch is a list of length batch_size, where each index contains what
#                              the dataset __getitem__ function returns.

#     Returns: To be compatible with the provided training loop, you should be returning
#         * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
#         * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
#         * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
#         * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
#         * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
#     '''
#     # TODO
#     return [], [], [], [], []

# def test_collate_fn(batch):
#     '''
#     Collation function to perform dynamic padding for inference on the test set.

#     Inputs:
#         * batch (List[Any]): batch is a list of length batch_size, where each index contains what
#                              the dataset __getitem__ function returns.

#     Recommended returns: 
#         * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
#         * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
#         * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
#     '''
#     # TODO
#     return [], [], []

# def get_dataloader(batch_size, split):
#     data_folder = 'data'
#     dset = T5Dataset(data_folder, split)
#     shuffle = split == "train"
#     collate_fn = normal_collate_fn if split != "test" else test_collate_fn

#     dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
#     return dataloader

# def load_t5_data(batch_size, test_batch_size):
#     train_loader = get_dataloader(batch_size, "train")
#     dev_loader = get_dataloader(test_batch_size, "dev")
#     test_loader = get_dataloader(test_batch_size, "test")
    
#     return train_loader, dev_loader, test_loader


# def load_lines(path):
#     with open(path, 'r') as f:
#         lines = f.readlines()
#         lines = [line.strip() for line in lines]
#     return lines

# def load_prompting_data(data_folder):
#     # TODO
#     return train_x, train_y, dev_x, dev_y, test_x



