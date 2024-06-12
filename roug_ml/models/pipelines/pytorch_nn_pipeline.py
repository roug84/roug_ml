import logging

log = logging.getLogger(__name__)  # noqa: E402

logging.basicConfig(level=logging.INFO)

from beartype.typing import Optional, Tuple
import torch
torch.cuda.empty_cache()  # Clear memory cache if needed
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)
from transformers import Trainer
import gc
gc.collect()  # Explicit garbage collection

import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from datasets import Dataset

import json
from sklearn.metrics import accuracy_score
import os
import numpy as np
import random
from transformers import PreTrainedTokenizer

from roug_ml.utl.evaluation.eval_utl import calc_loss_acc
from roug_ml.models.nn_models import (
    MLPTorchModel,
    CNNTorch,
    FlexibleCNN2DTorch,
    CNN2DTorch,
    FlexibleCNN2DTorchPower,
    MyTorchModel,
)

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    # AdamW,
    DataCollatorForLanguageModeling,
)

from torch.optim import AdamW


from roug_ml.utl.set_seed import set_seed

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(123)
random.seed(123)

# When using a GPU
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

set_seed(42)

# For PyTorch
torch.manual_seed(42)

# Also set the seed for the CUDA RNG if you're using a GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additional options for deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CustomTrainer(Trainer):
    def train(self, resume_from_checkpoint=None):
        # Override to add custom behavior or to handle training differently
        super().train(resume_from_checkpoint=resume_from_checkpoint)
        # Additional custom actions post training


class NNTorch:
    """
    Model to wrap around PyTorch neural networks.
    """

    def __init__(
        self,
        nn_key: str = "MLP",
        nn_params: dict = None,
        n_epochs: int = 10,
        batch_size: int = 100,
        learning_rate: float = 0.001,
        cost_function: torch.nn.Module = torch.nn.MSELoss(),
        metrics: list = ["accuracy"],
        tokenizer_name: str = 'gpt2',
        verbose: bool = True,
    ):
        """
        Instantiate the model.
        :param nn_key: The type of neural network to use.
        :param nn_params: Parameters for constructing the neural network.
        :param n_epochs: Number of epochs for training.
        :param batch_size: Batch size for training.
        :param learning_rate: Learning rate for the optimizer.
        :param cost_function: Loss function for optimization.
        :param metrics: List of evaluation metrics.
        :param tokenizer_name: The pretrained tokenizer model name.
        :param verbose: Whether to print training progress.
        """
        self.val_accuracies = None
        set_seed(42)
        self.nn_params = nn_params
        self.nn_key = nn_key
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_function = cost_function
        self.verbose = verbose
        self.metrics = metrics
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.nn_model = self.create_model()

    def create_model(self):
        """
        Create the neural network model based on the specified key.
        """

        nn_params = self.nn_params

        if self.nn_key == "MLP":
            return MLPTorchModel(
                input_shape=nn_params["input_shape"],
                output_shape=nn_params["output_shape"],
                in_nn=nn_params["in_nn"],
                activations=nn_params["activations"],
            )

        elif self.nn_key == "CNN":
            return CNNTorch(
                input_shape=nn_params["input_shape"],
                output_shape=nn_params["output_shape"],
                filters=nn_params["filters"],
                in_nn=nn_params["in_nn"],
                activations=nn_params["activations"],
            )

        elif self.nn_key == "CNN2D":
            return CNN2DTorch(
                input_shape=nn_params["input_shape"],
                output_shape=nn_params["output_shape"],
                conv_filters=nn_params["filters"],
                fc_nodes=nn_params["fc_nodes"],
                # activation=nn_params['activation']
            )

        elif self.nn_key == "FlexCNN2D":
            return FlexibleCNN2DTorch(
                input_shape=nn_params["input_shape"],
                output_shape=nn_params["output_shape"],
                conv_filters=nn_params["filters"],
                fc_nodes=nn_params["fc_nodes"],
                activation=nn_params["activation"],
            )

        elif self.nn_key == "AlexNet":
            layers_to_train = ["features.conv4", "classifier.dense2"]
            alexnet = torchvision.models.alexnet(pretrained=True)
            for name, param in alexnet.named_parameters():
                if all(not name.startswith(layer) for layer in layers_to_train):
                    param.requires_grad = False

            # Get the number of input features from the existing last layer
            num_ftrs = alexnet.classifier[6].in_features

            # Replace the final layer with a new one having the desired output size
            print(self.nn_params)  # Add this line for debugging

            alexnet.classifier[6] = torch.nn.Linear(
                num_ftrs, self.nn_params["output_shape"]
            )

            return alexnet

        elif self.nn_key == "GPt2Net":
            # Load pre-trained GPT-2 model and tokenizer
            # model_name = "gpt2"
            model_name = "gpt2"
            dropout = self.nn_params["dropout"]

            config = GPT2Config.from_pretrained(
                model_name, embd_pdrop=dropout, attn_pdrop=dropout
            )
            model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

            # Assuming training is handled separately as in your provided GPT-2 code
            return model

        elif self.nn_key == "FlexibleCNN2DTorchPower":

            return FlexibleCNN2DTorchPower(
                conv_module_params=nn_params["conv_module_params"],
                classifier_params=nn_params["classifier_params"],
                avgpool_output_size=nn_params["avgpool_output_size"],
            )
        elif self.nn_key == "my":
            return MyTorchModel(
                input_shape=nn_params["input_shape"],
                output_shape=nn_params["output_shape"],
                in_nn=nn_params["in_nn"],
                activations=nn_params["activations"],
            )

        else:
            raise ValueError(f"Unsupported nn_key: {self.nn_key}")

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            continue_training: bool = True,
            dataset: Optional[Dataset] = None,
            data_collator: Optional[PreTrainedTokenizer] = None,
            model_path: Optional[str] = None
    ) -> 'PyTorchNNPipeline':
        """
        Trains the neural network using the specified dataset or input arrays and targets.

        :param X: np.ndarray - Training input data, expected shape (n_samples, n_features).
        :param y: np.ndarray - Target values, expected shape (n_samples,) for regression or
                  (n_samples, n_classes) for classification.
        :param validation_data: Optional[Tuple[np.ndarray, np.ndarray]] - A tuple containing
                                validation data and its corresponding targets (X_val, y_val).
        :param continue_training: bool - Flag to determine whether to continue training from the
                                       last checkpoint or restart training.
        :param dataset: Optional[Dataset] - The dataset object from the Hugging Face `datasets`
                                             library, required for training models like GPT-2.
        :param data_collator: Optional[PreTrainedTokenizer] - The data collator to format the
                               data correctly for training, particularly necessary for
                               tokenization and batching of textual data when training models like GPT-2.
        :param model_path: Optional[str] - The file path where the trained model should be saved.
                                           It is essential for scenarios where the model needs to
                                           be saved incrementally or post-training.

        :return: Returns the instance itself, allowing for chaining or direct access to modified attributes.
        """

        if self.verbose:
            print(self.nn_model)

        if self.nn_key == "GPt2Net":
            if dataset is None or model_path is None:
                raise ValueError(
                    "Dataset and model_path are required for GPT-2 training."
                )
            # Define optimizer and scheduler
            optimizer = AdamW(self.nn_model.parameters(), lr=self.learning_rate)

            num_train_epochs = self.n_epochs
            num_training_steps = num_train_epochs * len(dataset)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
            )

            # Unfreeze the last layer
            for parameter in self.nn_model.transformer.h[-1].parameters():
                parameter.requires_grad = True

            # Training arguments
            training_args = TrainingArguments(
                output_dir=model_path,
                overwrite_output_dir=True,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=4,
                save_steps=10_000,
                save_total_limit=2,
            )

            # Initialize Trainer
            trainer = Trainer(
                model=self.nn_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
                optimizers=(optimizer, scheduler),
            )
            # Start fine-tuning
            trainer.train()

            # Save the trained model and tokenizer
            self.nn_model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)

            return self

        if not continue_training:
            self.nn_model = self.create_model()

        # torch.backends.cudnn.enabled = False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn_model.to(device)

        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.nn_model = torch.nn.DataParallel(self.nn_model)
        else:
            print(f"Not using GPUs!")

        # log.info(self.nn_model.parameters())
        optimizer_choice = optim.Adam(
            self.nn_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            amsgrad=False,
        )

        log.info("Optimizer: done")
        X, y = torch.Tensor(X), torch.Tensor(y)
        dataset = TensorDataset(X, y)
        log.info("Dataset created: done")
        # dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        log.info("Dataloader: done")
        if validation_data is not None:
            x_val, y_val = (
                torch.Tensor(validation_data[0]),
                torch.Tensor(validation_data[1]),
            )
            val_dataset = TensorDataset(x_val, y_val)
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )  # <--- HERE
            # Add a new list to store validation accuracies
            self.val_accuracies = []

        for epoch in range(self.n_epochs):
            # log.info("Start training")
            train_loss = 0.0
            train_correct = 0
            num_train_samples = 0

            if epoch == 50:
                for param_group in optimizer_choice.param_groups:
                    param_group["lr"] = 0.00001
                    print(param_group["lr"])

            if epoch == 60:
                for param_group in optimizer_choice.param_groups:
                    param_group["lr"] = 0.001
                    print(param_group["lr"])

            if epoch == 70:
                for param_group in optimizer_choice.param_groups:
                    param_group["lr"] = 0.0001
                    print(param_group["lr"])

            # Training
            self.nn_model.train()
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_choice.zero_grad()
                outputs = self.nn_model(inputs)

                loss = self.cost_function(outputs, targets)
                loss.backward()
                del loss

                optimizer_choice.step()

                batch_loss, batch_acc = calc_loss_acc(
                    outputs, targets, self.cost_function
                )
                train_loss = train_loss + batch_loss
                train_correct = train_correct + batch_acc * targets.size(
                    0
                )  # un-normalize the accuracy
                num_train_samples = num_train_samples + targets.size(0)

            train_loss = train_loss / len(dataloader)
            train_acc = train_correct / num_train_samples

            # log.info("Loss computation")
            # Validation
            if validation_data is not None:
                val_loss, val_acc = self.validate_model(
                    self.nn_model, val_dataloader, self.cost_function, device
                )
                self.val_accuracies.append(val_acc)

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs} Train Loss: "
                    f"{train_loss:.4f} Train Acc: {train_acc:.4f}",
                    end="",
                )
                if validation_data is not None:  # or validation_dataloader is not None:
                    print(f" Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
                else:
                    print()

        return self

    @staticmethod
    def validate_model(nn_model, val_dataloader, cost_function, device) -> tuple:
        """
        Validate the neural network model.

        :param nn_model: The neural network model.
        :param val_dataloader: Dataloader for validation data.
        :param cost_function: Loss function for evaluation.
        :param device: Device to perform validation.

        :return validation loss and accuracy.
        """
        nn_model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        num_train_samples = 0
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = nn_model(inputs)
            batch_loss, batch_acc = calc_loss_acc(outputs, targets, cost_function)
            val_loss += batch_loss
            val_correct += batch_acc * targets.size(0)  # un-normalize the accuracy
            num_train_samples += targets.size(0)

        val_loss /= len(val_dataloader)
        val_acc = val_correct / num_train_samples

        return val_loss, val_acc

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model's performance on the given data.

        :param X: Input data.
        :param y: Target values.

        :return:Score based on evaluation metric.
        """
        # Perform evaluation using the desired metric
        # Compute and return the score
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.
        :param deep: Whether to recursively fetch parameters.
        :return: Parameters for this estimator.
        """
        return self.__dict__

    def set_params(self, **params: dict) -> dict:
        """
        Set parameters for this estimator.
        :param params: parameter names and their new values.

        :return: the model with **param parameters
        """
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def save(self, model_path: str, model_name: str) -> None:
        """
        Saving the model.
        :param model_path: Path to save the model.
        :param model_name: Name of the model file.

        :return: None
        """
        if len(model_name) > 100:
            model_name = model_name[:100]

        torch.save(
            self.nn_model.state_dict(), os.path.join(model_path, model_name + ".pt")
        )
        # save additional params in a json
        params = {
            "nn_params": self.nn_params,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "cost_function": str(self.cost_function),
            "verbose": self.verbose,
            "metrics": self.metrics,
        }
        with open(os.path.join(model_path, model_name + "_params.json"), "w") as f:
            json.dump(params, f)

    def load(self, model_path: str, model_name: str) -> torch.nn.Module:
        """
        Load a model from disk.

        :param model_path: Path where the model is saved.
        :param model_name: Name of the model file.
        :return: the loaded model
        """
        # Shorten the model name if it's too long
        if len(model_name) > 100:
            model_name = model_name[:100]
        self.nn_model.load_state_dict(
            torch.load(os.path.join(model_path, model_name + ".pt"))
        )
        # load additional params from a json
        with open(os.path.join(model_path, model_name + "_params.json"), "r") as f:
            params = json.load(f)
        self.nn_params = params["nn_params"]
        self.n_epochs = params["n_epochs"]
        self.batch_size = params["batch_size"]
        self.learning_rate = params["learning_rate"]
        if params["cost_function"] == "MSELoss":
            self.cost_function = torch.nn.MSELoss()
        # add other cost functions as needed
        self.verbose = params["verbose"]
        self.metrics = params["metrics"]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.

        :param X: Input data.

        :return: Prediction.
        """
        self.nn_model.eval()  # Set the model to evaluation mode
        X = torch.Tensor(X)
        with torch.no_grad():  # Do not calculate gradients to speed up computation
            outputs = self.nn_model(X)
            probabilities = torch.nn.functional.softmax(
                outputs, dim=1
            )  # Apply softmax to get probabilities
            _, predicted_classes = torch.max(
                probabilities, 1
            )  # Get the class with the highest probability
        return predicted_classes.numpy()  # Convert tensor to numpy array

    def generate_text(self, input_ids: np.ndarray, max_length: int = 50) -> str:
        """
        Generate text using the GPT-2 model or any other suitable generative model.

        :param input_ids: Input token IDs for starting the text generation.
        :param max_length: Maximum length of the generated text sequence.
        :return: Generated text as a string.
        """
        self.nn_model.eval()  # Set the model to evaluation mode
        device = next(self.nn_model.parameters()).device  # Get the model's device

        # Convert input IDs to a tensor and send to the appropriate device
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.nn_model(input_ids)
                predictions = outputs[:, -1, :]  # get the last token logits
                predicted_index = torch.argmax(predictions, axis=-1).unsqueeze(0)  # get the most likely next token ID
                generated = torch.cat((generated, predicted_index), dim=1)  # append to the sequence
                input_ids = predicted_index

                if predicted_index.item() == self.tokenizer.eos_token_id:
                    break  # stop generating if the end of sequence token is generated

        generated_text = self.tokenizer.decode(generated.squeeze().tolist())  # decode the token IDs to text
        return generated_text
