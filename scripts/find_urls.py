import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
# Example: Save results to files
import pandas as pd
import torch

if __name__ == "__main__":
    """
    root = Path('/Users/pdomingo/Desktop/Doctorat/Datasets/laion_splits/')
    real_dataset = Path('/Users/pdomingo/Desktop/Doctorat/Datasets/laion_600k/')
    print('Open dataset...')
    dataset = load_dataset('arrow', data_files={'train': str(root.parent / 'improved_aesthetics_6.5plus-train.arrow')})

    print('Getting real_dataset_urls_text...')
    real_dataset_urls_text = set(tqdm(zip(dataset['train']['URL'], dataset['train']['TEXT'])))
    # Convert real_dataset_urls_text to a dictionary for faster lookup
    real_text_to_url = {text: url for url, text in real_dataset_urls_text}
    print('LEN - real_dataset_urls_text: ', len(real_dataset_urls_text))
    
    print('Getting downloaded_items_first_attempt...')
    downloaded_items_first_attempt = []
    for caption_path in tqdm(real_dataset.glob('**/*.txt')):
        item_id = caption_path.parent.stem
        with open(caption_path, 'r') as file:
            caption = file.read()
        downloaded_items_first_attempt.append((item_id, caption))
    print('LEN - downloaded_items_first_attempt: ', len(downloaded_items_first_attempt))

    print('Getting downloaded_first_only...')
    downloaded_first_only = []
    donwloaded_not_found = []
    for item_id, text in tqdm(downloaded_items_first_attempt, desc="Processing first download texts"):
        if text in real_text_to_url:  # If text exists in the real dataset
            url = real_text_to_url[text]
            downloaded_first_only.append((url, text, item_id))
        else:
            donwloaded_not_found.append((text, item_id))
    print('LEN - downloaded_first_only: ', len(downloaded_first_only))

    pd.DataFrame(downloaded_first_only, columns=['URL', 'TEXT', 'ITEM']).to_csv('downloaded_first_only.csv', index=False)

    """
    df = pd.read_csv('downloaded_first_only.csv')
    df['ITEM'] = df['ITEM'].apply(lambda x: str(x).zfill(6))
    real_dataset = Path('/Users/pdomingo/Desktop/Doctorat/Datasets/laion_600k/')
    # Load Dataset, Transforms, and Dataloaders
    items = [path.stem for path in real_dataset.iterdir()]
    [items.remove(val_item) for val_item in [
        '387721', '100120', '094499', '154839', '356719', '162689',
        '089960', '555388', '033550', '081209', '603533', '538392'
    ]]
    print('LEN - items: ', len(items))
    train_items, val_items, test_items = torch.utils.data.random_split(
        items, (0.801, 0.15, 1 - (0.801+0.15)),
        generator=torch.Generator(device='cpu').manual_seed(42)
    )

    train_set = set(train_items)
    val_set = set(val_items)
    test_set = set(test_items)

    # Filter the DataFrame for each partition
    train_df = df[df['ITEM'].isin(train_set)]
    val_df = df[df['ITEM'].isin(val_set)]
    test_df = df[df['ITEM'].isin(test_set)]

    # Save each partition to a separate CSV file
    train_df.to_csv('train_partition.csv', index=False)
    val_df.to_csv('val_partition.csv', index=False)
    test_df.to_csv('test_partition.csv', index=False)
    