"""Generate Morse code dataset for seq2static classification and seq2seq translation."""

import logging
from pathlib import Path
import re
import urllib.request

import h5py
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .morse_dict import (
        get_char_to_class,
        get_morse_dict,
        get_num_classes,
    )
except ImportError:
    from morse_dict import (
        get_char_to_class,
        get_morse_dict,
        get_num_classes,
    )


def download_tiny_shakespeare(cache_dir: Path | None = None) -> Path:
    """Download Tiny Shakespeare dataset from Karpathy's char-rnn repo.

    Args:
        cache_dir: Directory to cache the file. If None, uses current directory.

    Returns:
        Path to the downloaded text file.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = cache_dir / "tinyshakespeare.txt"

    if output_path.exists():
        logger.info(f"Using cached Tiny Shakespeare: {output_path}")
        return output_path

    logger.info(f"Downloading Tiny Shakespeare from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download Tiny Shakespeare: {e}") from e

    return output_path


def clean_text_for_morse(text: str) -> str:
    """Clean text to only include morse-compatible characters.

    Keeps only A-Z, 0-9, and space. Converts to uppercase, replaces
    all non-morse characters with spaces, and collapses multiple spaces.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string containing only A-Z, 0-9, and spaces.
    """
    # Convert to uppercase
    text = text.upper()

    # Replace all non-morse characters (not A-Z, 0-9, space) with space
    text = re.sub(r'[^A-Z0-9 ]', ' ', text)

    # Collapse multiple spaces to single space
    text = re.sub(r' +', ' ', text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text


def generate_morse_dataset(
    output_path: str,
    num_samples: int = 1000,
    steps_per_symbol: int = 3,
    on_value: float = 1.0,
    off_value: float = 0.0,
    include_numbers: bool = False,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    add_silence_between_symbols: bool = True,
    noise_std: float = 0.1,
) -> None:
    """Generate Morse code dataset for seq2static classification.

    Each sample is a sequence representing a single Morse code character (A-Z, optionally 0-9).
    The task is to classify the entire sequence into one of the character classes.

    Args:
        output_path: Path where the HDF5 file will be saved.
        num_samples: Total number of samples to generate.
        steps_per_symbol: Number of timesteps per symbol (dot, dash, or silence).
        on_value: Value to use when a channel is active (dot or dash).
        off_value: Value to use when a channel is inactive.
        include_numbers: If True, include numbers 0-9 in addition to letters A-Z.
        train_split: Fraction of samples for training (default 0.7).
        val_split: Fraction of samples for validation (default 0.15).
        test_split: Fraction of samples for testing (default 0.15).
        seed: Random seed for reproducibility.
        add_silence_between_symbols: If True, add silence between dots/dashes within a character.
        noise_std: Standard deviation of Gaussian noise to add to the sequences. Set to 0.0 to disable.

    Raises:
        ValueError: If splits don't sum to 1.0 or if num_samples is too small.
    """
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")

    # Validate num_samples
    num_classes = get_num_classes(include_numbers)
    min_samples = num_classes * 3  # Need at least 3 samples per class for splits
    if num_samples < min_samples:
        raise ValueError(
            f"num_samples ({num_samples}) too small. Need at least {min_samples} "
            f"to have samples for all {num_classes} classes across train/val/test splits."
        )

    # Setup random number generator
    rng = np.random.default_rng(seed)

    # Get Morse code dictionary and mappings
    morse_dict = get_morse_dict(include_numbers)
    char_to_class = get_char_to_class(include_numbers)
    characters = sorted(morse_dict.keys())

    # Calculate split sizes
    n_train = int(train_split * num_samples)
    n_val = int(val_split * num_samples)
    n_test = num_samples - n_train - n_val

    # Generate samples
    all_data = []
    all_labels = []

    # Balance samples across classes as much as possible
    samples_per_class = num_samples // num_classes
    extra_samples = num_samples % num_classes

    sample_count = 0
    for char_idx, char in enumerate(characters):
        # Determine how many samples for this class
        n_for_class = samples_per_class + (1 if char_idx < extra_samples else 0)

        for _ in range(n_for_class):
            if sample_count >= num_samples:
                break

            # Get Morse pattern for this character
            pattern = morse_dict[char]

            # Build sequence
            sequence = []

            for symbol_idx, symbol in enumerate(pattern):
                # Add the symbol (dot or dash)
                if symbol == '.':
                    # Dot: [on_value, off_value]
                    symbol_seq = np.full((steps_per_symbol, 2), [on_value, off_value], dtype=np.float32)
                elif symbol == '-':
                    # Dash: [off_value, on_value]
                    symbol_seq = np.full((steps_per_symbol, 2), [off_value, on_value], dtype=np.float32)
                else:
                    raise ValueError(f"Invalid symbol '{symbol}' in pattern '{pattern}'")

                sequence.append(symbol_seq)

                # Add silence between symbols (except after the last one)
                if add_silence_between_symbols and symbol_idx < len(pattern) - 1:
                    silence_seq = np.full((steps_per_symbol, 2), [off_value, off_value], dtype=np.float32)
                    sequence.append(silence_seq)

            # Stack into single array
            full_sequence = np.vstack(sequence)  # Shape: [T, 2]
            all_data.append(full_sequence)
            all_labels.append(char_to_class[char])

            sample_count += 1

    # Convert to numpy arrays
    # Note: sequences have variable lengths, so we'll pad to max length
    max_length = max(seq.shape[0] for seq in all_data)

    # Pad sequences to max length and add noise
    padded_data = []
    for seq in all_data:
        if seq.shape[0] < max_length:
            padding = np.full((max_length - seq.shape[0], 2), [off_value, off_value], dtype=np.float32)
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq.copy()

        # Add noise to the sequence
        if noise_std > 0.0:
            noise = rng.normal(0.0, noise_std, size=padded_seq.shape).astype(np.float32)
            padded_seq = padded_seq + noise

        padded_data.append(padded_seq)

    data_array = np.stack(padded_data, axis=0).astype(np.float32)  # Shape: [N, T, 2]
    labels_array = np.array(all_labels, dtype=np.int64)  # Shape: [N]

    # Shuffle data
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    data_array = data_array[indices]
    labels_array = labels_array[indices]

    # Split into train/val/test
    train_data = data_array[:n_train]
    train_labels = labels_array[:n_train]

    val_data = data_array[n_train:n_train + n_val]
    val_labels = labels_array[n_train:n_train + n_val]

    test_data = data_array[n_train + n_val:]
    test_labels = labels_array[n_train + n_val:]

    # Save to HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Train split
        train_group = f.create_group('train')
        train_group.create_dataset('data', data=train_data)
        train_group.create_dataset('labels', data=train_labels)

        # Validation split
        val_group = f.create_group('val')
        val_group.create_dataset('data', data=val_data)
        val_group.create_dataset('labels', data=val_labels)

        # Test split
        test_group = f.create_group('test')
        test_group.create_dataset('data', data=test_data)
        test_group.create_dataset('labels', data=test_labels)

    # Log summary
    logger.info(f"Generated Morse code dataset: {output_path}")
    logger.info(f"  Total samples: {num_samples}")
    logger.info(f"  Classes: {num_classes} ({'A-Z + 0-9' if include_numbers else 'A-Z'})")
    logger.info(f"  Train: {n_train} samples")
    logger.info(f"  Val: {n_val} samples")
    logger.info(f"  Test: {n_test} samples")
    logger.info(f"  Sequence shape: {data_array.shape}")
    logger.info(f"  Steps per symbol: {steps_per_symbol}")
    logger.info(f"  On/Off values: {on_value}/{off_value}")
    logger.info(f"  Silence between symbols: {add_silence_between_symbols}")
    logger.info(f"  Noise std: {noise_std}")


def generate_morse_translation_dataset(
    output_path: str,
    sequence_length: int = 15,
    num_samples: int = 1000,
    text_corpus_path: str | None = None,
    steps_per_symbol: int = 3,
    on_value: float = 1.0,
    off_value: float = 0.0,
    include_numbers: bool = False,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    add_silence_between_symbols: bool = True,
    noise_std: float = 0.1,
) -> None:
    """Generate Morse code dataset for seq2seq translation task.

    Each sample is a sequence of morse code representing multiple characters.
    The task is to translate morse code sequences back to text (character-level).
    Random subsequences are sampled from a text corpus (Tiny Shakespeare by default).

    Args:
        output_path: Path where the HDF5 file will be saved.
        sequence_length: Number of characters per text sample (e.g., 15).
        num_samples: Total number of samples to generate.
        text_corpus_path: Path to text file. If None, downloads Tiny Shakespeare.
        steps_per_symbol: Number of timesteps per symbol (dot, dash, or silence).
        on_value: Value to use when a channel is active (dot or dash).
        off_value: Value to use when a channel is inactive.
        include_numbers: If True, include numbers 0-9 in addition to letters A-Z.
        train_split: Fraction of samples for training (default 0.7).
        val_split: Fraction of samples for validation (default 0.15).
        test_split: Fraction of samples for testing (default 0.15).
        seed: Random seed for reproducibility.
        add_silence_between_symbols: If True, add silence between dots/dashes within a character.
        noise_std: Standard deviation of Gaussian noise to add to the sequences. Set to 0.0 to disable.

    Raises:
        ValueError: If splits don't sum to 1.0 or if corpus is too short.
    """
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")

    # Setup random number generator
    rng = np.random.default_rng(seed)

    # Load and clean text corpus
    if text_corpus_path is None:
        corpus_path = download_tiny_shakespeare()
    else:
        corpus_path = Path(text_corpus_path)

    with open(corpus_path, encoding='utf-8') as f:
        raw_text = f.read()

    cleaned_text = clean_text_for_morse(raw_text)

    if len(cleaned_text) < sequence_length:
        raise ValueError(
            f"Cleaned corpus length ({len(cleaned_text)}) is shorter than "
            f"sequence_length ({sequence_length}). Need at least {sequence_length} characters."
        )

    # Get morse dictionary and mappings (include space for translation)
    morse_dict = get_morse_dict(include_numbers=include_numbers, include_space=True)
    char_to_class = get_char_to_class(include_numbers=include_numbers, include_space=True)
    num_classes = get_num_classes(include_numbers=include_numbers, include_space=True)

    # Generate samples
    all_data = []
    all_labels = []

    max_possible_start = len(cleaned_text) - sequence_length

    for _ in range(num_samples):
        # Random starting position
        start_pos = rng.integers(0, max_possible_start + 1)
        text_sample = cleaned_text[start_pos:start_pos + sequence_length]

        # Convert to morse sequence with aligned labels
        morse_sequence = []
        label_sequence = []

        for char in text_sample:
            if char not in morse_dict:
                continue

            pattern = morse_dict[char]
            char_class = char_to_class[char]

            # Build morse pattern for this character
            for symbol_idx, symbol in enumerate(pattern):
                if symbol == '.':
                    symbol_seq = np.full((steps_per_symbol, 2), [on_value, off_value], dtype=np.float32)
                elif symbol == '-':
                    symbol_seq = np.full((steps_per_symbol, 2), [off_value, on_value], dtype=np.float32)
                elif symbol == ' ':  # Space character (7 silence steps)
                    symbol_seq = np.full((steps_per_symbol, 2), [off_value, off_value], dtype=np.float32)
                else:
                    raise ValueError(f"Invalid symbol '{symbol}' in pattern '{pattern}'")

                morse_sequence.append(symbol_seq)
                # Repeat label for each timestep in this symbol
                label_sequence.extend([char_class] * steps_per_symbol)

                # Add silence between symbols (except after the last one)
                if add_silence_between_symbols and symbol_idx < len(pattern) - 1:
                    silence_seq = np.full((steps_per_symbol, 2), [off_value, off_value], dtype=np.float32)
                    morse_sequence.append(silence_seq)
                    # Repeat label during silence between symbols
                    label_sequence.extend([char_class] * steps_per_symbol)

            # Add silence between characters (3 timesteps by convention)
            if add_silence_between_symbols:
                char_silence = np.full((steps_per_symbol, 2), [off_value, off_value], dtype=np.float32)
                morse_sequence.append(char_silence)
                # Repeat last char label during inter-char silence
                label_sequence.extend([char_class] * steps_per_symbol)

        if not morse_sequence:
            continue

        morse_array = np.vstack(morse_sequence)  # Shape: [T, 2]
        labels_array = np.array(label_sequence, dtype=np.int64)  # Shape: [T]

        # Verify lengths match
        assert morse_array.shape[0] == len(labels_array), f"Morse timesteps {morse_array.shape[0]} != label count {len(labels_array)}"

        # Add noise
        if noise_std > 0.0:
            noise = rng.normal(0.0, noise_std, size=morse_array.shape).astype(np.float32)
            morse_array = morse_array + noise

        all_data.append(morse_array)
        all_labels.append(labels_array)

    # Pad sequences to max length
    if not all_data:
        raise ValueError("No valid samples generated. Check corpus length and sequence_length.")

    max_length = max(seq.shape[0] for seq in all_data)
    actual_num_samples = len(all_data)

    padded_data = []
    padded_labels = []

    for morse_seq, label_seq in zip(all_data, all_labels, strict=False):
        # Ensure both sequences have the same length
        assert morse_seq.shape[0] == label_seq.shape[0], f"Morse seq length {morse_seq.shape[0]} != label seq length {label_seq.shape[0]}"

        if morse_seq.shape[0] < max_length:
            # Pad morse with silence
            pad_length = max_length - morse_seq.shape[0]
            morse_padding = np.full((pad_length, 2), [off_value, off_value], dtype=np.float32)
            padded_morse = np.vstack([morse_seq, morse_padding])

            # Pad labels with -100 (ignore_index for cross_entropy)
            label_padding = np.full((pad_length,), -100, dtype=np.int64)
            padded_label = np.concatenate([label_seq, label_padding])
        else:
            padded_morse = morse_seq.copy()
            padded_label = label_seq.copy()

        # Verify lengths match
        assert padded_morse.shape[0] == max_length, f"Padded morse length {padded_morse.shape[0]} != max_length {max_length}"
        assert padded_label.shape[0] == max_length, f"Padded label length {padded_label.shape[0]} != max_length {max_length}"

        padded_data.append(padded_morse)
        padded_labels.append(padded_label)

    data_array = np.stack(padded_data, axis=0).astype(np.float32)  # Shape: [N, T, 2]
    labels_array = np.stack(padded_labels, axis=0).astype(np.int64)  # Shape: [N, T]

    # Update num_samples to actual number generated (in case some were skipped)
    num_samples = actual_num_samples

    # Shuffle data
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    data_array = data_array[indices]
    labels_array = labels_array[indices]

    # Split into train/val/test
    n_train = int(train_split * num_samples)
    n_val = int(val_split * num_samples)
    n_test = num_samples - n_train - n_val

    train_data = data_array[:n_train]
    train_labels = labels_array[:n_train]

    val_data = data_array[n_train:n_train + n_val]
    val_labels = labels_array[n_train:n_train + n_val]

    test_data = data_array[n_train + n_val:]
    test_labels = labels_array[n_train + n_val:]

    # Save to HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Train split
        train_group = f.create_group('train')
        train_group.create_dataset('data', data=train_data)
        train_group.create_dataset('labels', data=train_labels)

        # Validation split
        val_group = f.create_group('val')
        val_group.create_dataset('data', data=val_data)
        val_group.create_dataset('labels', data=val_labels)

        # Test split
        test_group = f.create_group('test')
        test_group.create_dataset('data', data=test_data)
        test_group.create_dataset('labels', data=test_labels)

    # Log summary
    logger.info(f"Generated Morse code translation dataset: {output_path}")
    logger.info(f"  Total samples: {num_samples}")
    logger.info(f"  Characters per sample: {sequence_length}")
    logger.info(f"  Classes: {num_classes} ({'A-Z + 0-9 + space' if include_numbers else 'A-Z + space'})")
    logger.info(f"  Train: {n_train} samples")
    logger.info(f"  Val: {n_val} samples")
    logger.info(f"  Test: {n_test} samples")
    logger.info(f"  Data shape: {data_array.shape}")
    logger.info(f"  Labels shape: {labels_array.shape}")
    logger.info(f"  Steps per symbol: {steps_per_symbol}")
    logger.info(f"  On/Off values: {on_value}/{off_value}")
    logger.info(f"  Silence between symbols: {add_silence_between_symbols}")
    logger.info(f"  Noise std: {noise_std}")
    logger.info("  Task: seq2seq classification (morse → text translation)")

