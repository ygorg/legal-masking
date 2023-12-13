from my_tokenize import initialize_tokenizer
from data_setup import create_small_datasets
from data_collator_setup import initialize_data_collator, demonstrate_data_collator
from model_trainer import initialize_model_and_trainer, downsample_dataset, train_and_evaluate
from my_tokenize import tokenize_function, group_texts
import argparse

def main():

    # Set up argument parser
# =============================================================================
    parser = argparse.ArgumentParser(description="Run the model training and evaluation")
    parser.add_argument("--batch_size", type=int, default=16, help="Training/Evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--version", type=int, default=1, help="Model version")
#     parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
# 
#     # Parse arguments
    args = parser.parse_args()
# =============================================================================

    data_dir = './data'  # Specify your data directory
    small_datasets = create_small_datasets(data_dir, 100)
    
    if small_datasets:
        tokenizer = initialize_tokenizer()
        chunk_size = 128  # Specify the chunk size
        
        # Tokenize datasets
        tokenized_datasets = {}
        for split, dataset in small_datasets.items():
            tokenized_datasets[split] = dataset.map(
                lambda examples: tokenize_function(tokenizer, examples),
                batched=True,
                remove_columns=['id', 'text', 'sector', 'descriptor', 'year', '__index_level_0__']
            )
        
        # Group texts
        for split in tokenized_datasets.keys():
            tokenized_datasets[split] = tokenized_datasets[split].map(
                lambda examples: group_texts(chunk_size, examples),
                batched=True
            )
        
        # Example output
        print("====================================================================")
        print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
        print(tokenizer.decode(tokenized_datasets["train"][0]["labels"]))

        # Initialize and demonstrate Data Collator
        data_collator = initialize_data_collator(tokenizer)
        demonstrate_data_collator(data_collator, tokenized_datasets, tokenizer)
    
        # Initialize the trainer with the model, datasets, and data collator
        #trainer = initialize_model_and_trainer(tokenized_datasets, data_collator, tokenizer)
        # Initialize the trainer with the model, datasets, data collator, and command line parameters
        version = 2
        batch_size = 8
        num_epochs = 2
        trainer = initialize_model_and_trainer(
            tokenized_datasets, 
            data_collator, 
            tokenizer, 
            version=args.version, 
            batch_size=args.batch_size, 
            num_epochs=args.num_epochs
        )
        # Add training code here if needed
        eval_results = train_and_evaluate(trainer)

if __name__ == "__main__":
    main()

