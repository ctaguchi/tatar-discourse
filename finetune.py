from datasets import (load_dataset,
                      Dataset,
                      IterableDataset,
                      Audio,
                      disable_caching,
                      Features,
                      Value,
                      Sequence)
from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor,
                          Wav2Vec2ProcessorWithLM,
                          Wav2Vec2ForCTC,
                          TrainingArguments,
                          Trainer)
from pyctcdecode import build_ctcdecoder
import torch
import jiwer
import numpy as np
import os
import json
import math
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import argparse
from dotenv import load_dotenv

# local
from utils import text_normalizer # TODO: replace text normalization-related functions with this

load_dotenv()


def load_data(dataset: str,
              lang: str,
              num_train_samples: Union[int, str],
              num_valid_samples: Union[int, str],
              cache_dir: str,
              enable_caching: bool,
              streaming: bool,
              hf_token: Optional[str]) -> Dict[str, Union[Dataset, IterableDataset]]:
    """Load data from Hugging Face Datasets.
    
    Args:
        dataset (str): Dataset name.
        lang (str): Language code.
        num_train_samples (int | str): The number of training samples.
        num_valid_samples (int | str): The number of validation samples.
        cache_dir (str): Cache directory.
        enable_caching (bool): If True, caching is enabled.
        streaming (bool): If True, the dataset is loaded in the streaming mode.
        hf_token (Optional[str]): Hugging Face token.
    """
    if not enable_caching:
        print("Caching disabled; redownloading the data...")
        download_mode = "force_redownload"
    else:
        print("Caching enabled; using the cached data...")
        download_mode = "reuse_dataset_if_exists" # default
        
    train = load_dataset(dataset,
                         lang,
                         split="train",
                         trust_remote_code=True,
                         cache_dir=cache_dir,
                         download_mode=download_mode,
                         streaming=streaming,
                         token=hf_token)
    valid = load_dataset(dataset,
                         lang,
                         split="validation",
                         trust_remote_code=True,
                         cache_dir=cache_dir,
                         download_mode=download_mode,
                         streaming=streaming,
                         token=hf_token)

    if isinstance(num_train_samples, int):
        train = train.take(num_train_samples)
    elif num_train_samples == "all":
        pass
    else:
        raise ValueError(f"Invalid num_train_samples: {num_train_samples}")

    if isinstance(num_valid_samples, int):
        valid = valid.take(num_valid_samples)
    elif num_valid_samples == "all":
        pass
    else:
        raise ValueError(f"Invalid num_valid_samples: {num_valid_samples}")
        
    return {"train": train,
            "valid": valid}


def batch_normalize_text(batch: dict,
                         lang: str) -> dict:
    """Normalize a batch of texts.
    
    Args:
        batch (dict): A batch of texts.
        lang (str): Language code.

    Returns:
        dict: A batch of normalized texts.
    """
    batch["sentence"] = text_normalizer.normalize_text(batch["sentence"],
                                                       lang=lang)
    return batch


def preprocess_audio(train: Union[Dataset, IterableDataset],
                     valid: Union[Dataset, IterableDataset],
                     sampling_rate: int = 16_000) -> Tuple[Dataset, Dataset]:
    """Preprocess the audio of the dataset."""
    train = train.cast_column("audio",
                              Audio(sampling_rate=sampling_rate))
    valid = valid.cast_column("audio",
                              Audio(sampling_rate=sampling_rate))
    return train, valid


def extract_all_chars(batch: dict) -> Dict[str, list]:
    """Extract all character types that appear in the dataset.
    This is for creating the vocabulary to be used in the training.
    """
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab],
            "all_text": [all_text]}


def create_vocab_for_iterabledataset(train: IterableDataset,
                                     valid: IterableDataset,
                                     vocab_file: str) -> int:
    """Prepare the vocabulary (character types) for training.
    
    Args:
        train (IterableDataset): Training dataset.
        valid (IterableDataset): Validation dataset.
        vocab_file (str): Vocabulary file name.

    Returns:
        int: The number of training samples. (necessary for estimating max_steps later)
    """
    assert isinstance(train, IterableDataset)
    assert isinstance(valid, IterableDataset)

    vocab = set()
    # print(train)
    for num_train_samples, sample in enumerate(train):
        # `sample`: dict
        char_types = set(sample["sentence"])
        vocab |= char_types
    for sample in valid:
        char_types = set(sample["sentence"])
        vocab |= char_types

    vocab_dict = {v: k for k, v in enumerate(vocab)}

    if " " not in vocab_dict:
        vocab_dict[" "] = len(vocab_dict)

    # to make it clear that a whitespace " " has its own token class,
    # we give it a more visible character "|".
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_file, "w") as f:
        json.dump(vocab_dict, f)

    return num_train_samples
        

def create_vocab(train: Union[Dataset, IterableDataset],
                 valid: Union[Dataset, IterableDataset],
                 vocab_file: str) -> None:
    """Prepare the vocabulary (character types) for training."""
    if isinstance(train, IterableDataset):
        vocab_train = train.map(extract_all_chars,
                                remove_columns=train.column_names)
    else:
        vocab_train = train.map(extract_all_chars,
                                batched=True,
                                batch_size=-1,
                                keep_in_memory=True,
                                remove_columns=train.column_names)
    if isinstance(valid, IterableDataset):
        vocab_valid = valid.map(extract_all_chars,
                                remove_columns=valid.column_names)
    else:
        vocab_valid = valid.map(extract_all_chars,
                                batched=True,
                                batch_size=-1,
                                keep_in_memory=True,
                                remove_columns=valid.column_names)
    vocab_list = list(
        set(vocab_train["vocab"][0]) | set(vocab_valid["vocab"][0])
    )
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    if " " not in vocab_dict:
        vocab_dict[" "] = len(vocab_dict)

    # to make it clear that a whitespace " " has its own token class,
    # we give it a more visible character "|".
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_file, "w") as f:
        json.dump(vocab_dict, f)


def prepare_dataset(batch: dict,
                    processor: Wav2Vec2Processor,
                    lang: str = "tt") -> dict:
    """Prepare the dataset for the training.
    Add `input_values` and `labels` to the dataset.
    """
    audio = batch["audio"]

    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0] # batched output is un-batched

    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = text_normalizer.normalize_text(batch["sentence"],
                                                     lang=lang)

    return batch


def debug_invalid_values(batch: dict):
    """Check if there is any invalid values in the input."""
    input_values = batch["input_values"]
    labels = batch["labels"]

    if isinstance(input_values, list):
        input_values = torch.tensor(input_values)
    
    if torch.isnan(input_values).any() or torch.isinf(input_values).any():
        raise ValueError(f"Invalid audio inputs detected. {batch['input_values']}")

    return batch

# compute_metrics is moved inside the main function

Feature = Dict[str, Union[List[int], torch.Tensor]]

@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set, it will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self,
                 features: List[Feature]) -> Dict[str, torch.Tensor]:
        """Split inputs and labels since they have to be of different lengths
        and need different padding methods
        """
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        # The syntax below is deprecated
        # label_features = [
        #     {"input_ids": feature["labels"]} for feature in features
        # ]
        
        label_texts = [feature["labels"] for feature in features]
        # print("Label texts:", label_texts)
        # ^ for debugging

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        labels_batch = self.processor(
            text=label_texts,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )
        
        batch["labels"] = labels

        return batch


def load_test(dataset: str,
              num_samples: Union[int, str],
              lang: str,
              cache_dir: str,
              download_mode: bool,
              streaming: bool,
              hf_token: Optional[str]) -> Dataset:
    """Load the test data.
    
    Args:
        dataset (str): Dataset name.
        num_samples (int | str): The number of test samples.
        lang (str): Language code.
        cache_dir (str): Cache directory.
        download_mode (bool): If True, the data is redownloaded.
        streaming (bool): If True, the dataset is loaded in the streaming mode.
        hf_token (Optional[str]): Hugging Face token.
    """
    test = load_dataset(dataset,
                        lang,
                        split="test",
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                        download_mode=download_mode,
                        streaming=streaming,
                        token=hf_token)
    
    if isinstance(num_samples, int):
        test = test.take(num_samples)
    elif num_samples == "all":
        pass
    else:
        raise ValueError(f"Invalid num_samples: {num_samples}")
    
    return test


def batch_transcribe(batch: dict,
                     model: Wav2Vec2ForCTC,
                     processor: Wav2Vec2Processor,
                     device: str) -> Dict[str, str]:
    """Transcribe audio."""
    inputs = processor(batch["input_values"], 
                       sampling_rate=16_000,
                       return_tensors="pt",
                       padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values,
                       attention_mask=inputs.attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1)

    batch["pred_str"] = processor.batch_decode(pred_ids,
                                               skip_special_tokens=True,
                                               group_tokens=True)[0]

    batch["text"] = batch["labels"]
    return batch


def test(model,
         processor,
         args: argparse.Namespace) -> None:
    """Evaluate on the test data.
    
    Args:
        model: Wav2Vec2ForCTC
        processor: Wav2Vec2Processor
        args: argparse.Namespace

    Returns:
        None
    """
    if not args.enable_caching:
        print("Caching disabled; redownloading the data...")
        download_mode = "force_redownload"
    else:
        print("Caching enabled; using the cached data...")
        download_mode = "reuse_dataset_if_exists" # default

    hf_token = os.getenv("HF_TOKEN")

    if args.num_test_samples is None:
        args.num_test_samples = args.num_valid_samples 
    
    test = load_test(args.dataset,
                     num_samples=args.num_valid_samples,
                     lang=args.lang,
                     cache_dir=args.cache_dir,
                     download_mode=download_mode,
                     streaming=args.streaming,
                     token=hf_token)
                    
    test = test.map(batch_normalize_text,
                    fn_kwargs={"lang": args.lang},
                    features=test.info.features)

    test = test.cast_column("audio",
                            Audio(sampling_rate=16_000))

    # TODO: we can probably move these Features variables to global
    ASRSampleFeatures = Features(
        {"input_values": Sequence(
            feature=Value(dtype="float32", id=None),
            length=-1,
            id=None
            ),
         "input_length": Value(dtype="int64", id=None),
         "labels": Value(dtype="string", id=None)}
        )
    
    ASRResultFeatures = Features(
        {"pred_str": Value(dtype="string", id=None),
         "text": Value(dtype="string", id=None)}
        )
    
    test = test.map(prepare_dataset,
                    remove_columns=test.column_names,
                    features=ASRSampleFeatures)

    results = test.map(batch_transcribe,
                       fn_kwargs={"model": model,
                                  "processor": processor,
                                  "device": args.device},
                       remove_columns=test.column_names,
                       features=ASRResultFeatures)
    
    test_cer = jiwer.cer(
        reference=results["text"],
        hypothesis=results["pred_str"],
    )
    print("Test CER:", test_cer)

    print(f"Test CER saved in: {args.test_cer_file}")


class DebugTrainer(Trainer):
    """For debugging purposes."""
    def compute_loss(self,
                     model,
                     inputs,
                     return_outputs=False,
                     num_items_in_batch=None):
        # Forward pass
        print("Input:")
        print(inputs)
        
        outputs = model(**inputs)

        # Inspect the output
        print("Model output:")
        print(outputs)

        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        return (loss, outputs) if return_outputs else loss


def get_args() -> argparse.Namespace:
    """Argument parser."""
    def sampletype(value: Union[int, str]):
        try:
            return int(value)
        except ValueError:
            if value == "all":
                return value
            else:
                raise argparse.ArgumentTypeError("Invalid sample type.")
            
    parser = argparse.ArgumentParser("Fine-tune Wav2Vec2.")
    parser.add_argument("-m",
                        "--model",
                        choices=[
                            "facebook/wav2vec2-large-xlsr-53",
                            "facebook/wav2vec2-base-960h",
                            "facebook/wav2vec2-base",
                            "facebook/wav2vec2-xls-r-300m",
                            "facebook/wav2vec2-xls-r-1b",
                            "facebook/wav2vec2-xls-r-2b",
                            "facebook/mms-300m",
                            "facebook/mms-1b",
                            "facebook/mms-1b-l1107",
                            "facebook/mms-1b-all",
                            ],
                        default="facebook/wav2vec2-large-xlsr-53",
                        help="Model to be used in fine-tuning.",
                        )
    parser.add_argument("-d",
                        "--dataset",
                        type=str,
                        default="mozilla-foundation/common_voice_17_0",
                        help="Dataset repo path in Hugging Face Datasets.",
                        )
    parser.add_argument("-l",
                        "--lang",
                        type=str,
                        help="Target language code.",
                        )
    parser.add_argument("-v",
                        "--vocab_file",
                        type=str,
                        default="vocab.json",
                        help="Vocab file name."
                        )
    parser.add_argument("-n",
                        "--num_proc",
                        type=int,
                        default=4,
                        help="The number of CPU processors for batched processing.",
                        )
    parser.add_argument("--num_train_samples",
                        type=sampletype,
                        default=5000,
                        help="The number of training samples.",
                        )
    parser.add_argument("--num_valid_samples",
                        type=sampletype,
                        default=500,
                        help="The number of validation samples.",
                        )
    parser.add_argument("--num_test_samples",
                        type=sampletype,
                        default=None,
                        help="The number of test samples.",
                        )
    parser.add_argument("--keep_punctuation",
                        action="store_true",
                        help="If True, punctuation and other special symbols will be kept.",
                        )
    parser.add_argument("--attention_dropout",
                        type=float,
                        default=0.1,
                        help="Attention dropout probability.",
                        )
    parser.add_argument("--hidden_dropout",
                        type=float,
                        default=0.1,
                        help="Hidden dropout probability.",
                        )
    parser.add_argument("--feat_proj_dropout",
                        type=float,
                        default=0.0,
                        help="Feature projection dropout probability.",
                        )
    parser.add_argument("--mask_time_prob",
                        type=float,
                        default=0.05,
                        help="The percentage of the whole axis (between 0 and 1) which will be masked.",
                        )
    parser.add_argument("--layerdrop",
                        type=float,
                        default=0.1,
                        help="Layerdrop probability.",
                        )
    parser.add_argument("--ctc_loss_reduction",
                        type=str,
                        default="mean",
                        help="CTC loss reduction method.",
                        )
    parser.add_argument("-o",
                        "--output_dir",
                        type=str,
                        default="models",
                        help="Output directory's path for the fine-tuned model.",
                        )
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=8,
                        help="Per-device batch size. Increase it if it seems to have some more room for better efficiency.",
                        )
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=2,
                        help="Gradient accumulation steps during the training.",
                        )
    parser.add_argument("--eval_strategy",
                        type=str,
                        default="steps",
                        help="Evaluation strategy.",
                        )
    parser.add_argument("-e",
                        "--num_train_epochs",
                        type=int,
                        default=30,
                        help="Number of training epochs.",
                        )
    parser.add_argument("--save_steps",
                        type=int,
                        default=100,
                        help="Frequency of saving a model checkpoint.",
                        )
    parser.add_argument("--eval_steps",
                        type=int,
                        default=100,
                        help="Frequency of evaluating the model.",
                        )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=10,
                        help="Frequency of logging the training stats",
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-5,
                        help="Learning rate of the training.",
                        )
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=500,
                        help="Warmup steps.",
                        )
    parser.add_argument("--save_total_limit",
                        type=int,
                        default=2,
                        help="Total limit number of the saved checkpoints at one time.",
                        )
    parser.add_argument("--kana",
                        action="store_true",
                        help="If True, Japanese will be converted to kana.",
                        )
    parser.add_argument("--cache_dir",
                        type=str,
                        default=".cache",
                        help="Cache directory for the loaded data.",
                        )
    parser.add_argument("--enable_caching",
                        action="store_true",
                        help="Enable caching of the loaded data.",
                        )
    parser.add_argument("--streaming",
                        action="store_true",
                        help="If True, the dataset is loaded in the streaming mode.",
                        )
    parser.add_argument("--report_to",
                        default="wandb",
                        type=str,
                        help="If `wandb`, the log is reported to wandb."
                        )
    parser.add_argument("--wandb_run_name",
                        default=None,
                        type=str,
                        help="You may optionally specify the run name for your wandb report.",
                        )
    parser.add_argument("--wandb_project",
                        default=None,
                        type=str,
                        help="Specify the project name for the wandb report.",
                        )
    parser.add_argument("--device",
                        default="cuda",
                        type=str,
                        help="Device name (GPU).",
                        )
    parser.add_argument("--gradient_checkpointing",
                        action="store_true",
                        help="If true, gradient checkpointing will be used.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="If true, it will be run in the debug mode.")
    parser.add_argument("--max_input_length",
                        type=float,
                        default=15.0,
                        help="Maximum audio sample length (in seconds)")
    parser.add_argument("--full_precision",
                        action="store_true",
                        help=(
                            "If specified, training will be done in full precision (fp32) rather than mixed precision. "
                            "This is necessary because mixed-precision training can result in numerical instability.")
                        )
    parser.add_argument("--test_cer_file",
                        type=str,
                        help="File path for saving the test CER score.")
    parser.add_argument("--use_lm",
                        action="store_true",
                        help="If true, a language model will be used for decoding.")
    parser.add_argument("--lm_path",
                        type=str,
                        help="Path to the language model.")
    
    return parser.parse_args()
        

if __name__ == "__main__":
    args = get_args()

    if not args.enable_caching:
        # disable caching to avoid OS Error File Too Large
        disable_caching()

    if args.report_to and args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # debug
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    hf_token = os.getenv("HF_TOKEN")
        
    datasets = load_data(args.dataset,
                         args.lang,
                         args.num_train_samples,
                         args.num_valid_samples,
                         args.cache_dir,
                         args.enable_caching,
                         args.streaming,
                         hf_token=hf_token)
    train = datasets["train"]
    valid = datasets["valid"]

    print("Dataset loaded.")

    original_features = train.info.features

    if args.debug:
        print("Preview:")
        if args.streaming:
            for sample in train:
                print(sample)
                break
            for sample in valid:
                print(sample)
                break
        else:
            print(train[0])
            print(valid[0])

    if not args.keep_punctuation:
        train = train.map(batch_normalize_text,
                          fn_kwargs={"lang": args.lang},
                        #   features=original_features
                        #   features=train.info.features
                          ) # necessary for IterableDataset to keep features
        # see this: https://github.com/huggingface/datasets/pull/5311
        valid = valid.map(batch_normalize_text,
                          fn_kwargs={"lang": args.lang},
                        #   features=original_features
                        #   features=valid.info.features
                          )


    train.info.features = original_features
    valid.info.features = original_features
    # print("train info features:", train.info.features)
        
    # for sample in train:
    #     print(sample)
    #     break

    print("Creating the vocab file...")
    vocab_file = os.path.join(args.output_dir,
                              args.vocab_file)
    if args.streaming:
        num_train_samples = create_vocab_for_iterabledataset(train,
                                                             valid,
                                                             vocab_file)
        print("Number of training samples:", num_train_samples)
        print("Vocab created from an iterable dataset.")
    else:
        create_vocab(train,
                     valid,
                     vocab_file)
        print("Vocab created.")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    print("Tokenizer created.")
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    print("Feature extractor created.")
    
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    print("Processor created.")

    if args.use_lm:
        # Use n-gram language model for decoding
        print("Set to use language model for decoding.")
        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab_dict = {
            k.lower(): v
            for k, v in sorted(vocab_dict.items(), key=lambda x: x[1])
            }
        
        print("Building the decoder...")
        decoder = build_ctcdecoder(
            labels=list(sorted_vocab_dict.keys()),
            kenlm_model_path=args.lm_path,
        )

        processor = Wav2Vec2ProcessorWithLM(
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            decoder=decoder,
        )


    train, valid = preprocess_audio(train, valid)
    print("Audio preprocessed.")

    print("Preparing the dataset...")
    if args.streaming:
        # Streaming mode (IterableDataset) does not support `num_proc`.
        new_features = Features(
            {"input_values": Sequence(
                feature=Value(dtype="float32", id=None),
                length=-1,
                id=None
                ),
             "input_length": Value(dtype="int64", id=None),
             "labels": Value(dtype="string", id=None)}
             )
        train = train.map(
            prepare_dataset,
            remove_columns=train.column_names,
            # features=new_features,
            fn_kwargs={"processor": processor,
                       "lang": args.lang},
            )
        valid = valid.map(
            prepare_dataset,
            remove_columns=valid.column_names,
            # features=new_features,
            fn_kwargs={"processor": processor,
                       "lang": args.lang},
            )
        train.info.features = new_features
        valid.info.features = new_features
    else:
        train = train.map(
            prepare_dataset,
            remove_columns=train.column_names,
            num_proc=args.num_proc,
            fn_kwargs={"processor": processor,
                       "lang": args.lang},
        )
        valid = valid.map(
            prepare_dataset,
            remove_columns=valid.column_names,
            num_proc=args.num_proc,
            fn_kwargs={"processor": processor,
                       "lang": args.lang},
        )

    def not_too_long(x: float) -> bool:
        """For filtering out audio samples that are too long."""
        return x < args.max_input_length * processor.feature_extractor.sampling_rate
    
    if args.max_input_length:
        train = train.filter(not_too_long,
                             input_columns=["input_length"])
    
    print("Dataset prepared")

    # debug
    if args.debug:
        train = train.map(debug_invalid_values,
                          features=train.info.features)
        valid = valid.map(debug_invalid_values,
                          features=valid.info.features)

    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
    )
    print("Data collator created.")

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=args.feat_proj_dropout,
        mask_time_prob=args.mask_time_prob,
        layerdrop=args.layerdrop,
        ctc_loss_reduction=args.ctc_loss_reduction,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    print("Model loaded")

    model.freeze_feature_extractor()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    fp16 = False if args.full_precision else True
    # if fp16 = True, the Trainer will use mixed precision
    log_level = "debug" if args.debug else "passive"

    if args.streaming:
        if args.num_train_samples == "all":
            args.num_train_samples = num_train_samples
        max_steps = math.ceil(args.num_train_samples / args.batch_size) * args.num_train_epochs

    # args.num_train_epochs is ignored when max_steps is specified.
    group_by_length = False if args.streaming else True
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=group_by_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        fp16=fp16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True, # to save the best model
        log_level=log_level,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.wandb_run_name,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        )

    def compute_metrics(pred) -> Dict[str, float]:
        """Compute the evaluation score (CER)."""
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics                                                                                                                                                       
        label_str = processor.batch_decode(pred.label_ids,
                                           group_tokens=False)
        
        cer = jiwer.cer(
            reference=label_str,
            hypothesis=pred_str
        )
        return {"cer": cer}

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=processor.feature_extractor,
    )

    print("Starting the training...")
    trainer.train()
    print("Training completed.")

    print(f"Saving the best model in {args.output_dir}...")
    trainer.save_model(args.output_dir)

    print(f"Saving the processor in {args.output_dir}")
    processor.save_pretrained(args.output_dir)

    print("Starting the evaluation...")
    test(model, processor, args)
