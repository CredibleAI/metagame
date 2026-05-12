import numpy as np
import torch

from shapiq import Game


class VisionLanguageGame(Game):
    """
    A general interface for Huggingface CLIP, SigLIP
    """
    def __init__(self, model, processor, input_image, input_text, batch_size=1, verbose=False):
        self.model = model
        self.model_type = "clip" 
        if "siglip2" in model.name_or_path:
            self.model_type = "siglip2"
        elif "siglip" in model.name_or_path:
            self.model_type = "siglip"
        self.processor = processor
        self.input_image = input_image
        self.input_text = input_text
        self.batch_size = batch_size
        self._force_float32 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7

        self.inputs = self._processor_function(input_image, input_text)

        self.image_size = model.vision_model.embeddings.image_size
        self.patch_size = model.vision_model.embeddings.patch_size
        self.n_channels = model.vision_model.embeddings.config.num_channels
        self.grid_size = self.image_size // self.patch_size
        self.n_players_image = int(self.image_size / self.patch_size) ** 2 

        # remove the bos and eos tokens
        if self.model_type == "siglip2":
            self.n_players_text = self.inputs.input_ids.count_nonzero().item() - 1 
        elif self.model_type == "siglip": 
            self.n_players_text = (self.inputs.input_ids != 1).count_nonzero().item()
        elif self.model_type == "clip": 
            self.n_players_text = self.inputs.input_ids.size(1) - 2 

        # get the normalization value
        coalitions = np.zeros((2, self.n_players_image + self.n_players_text), dtype=bool)

        coalitions[1, :] = True
        game_output = self.value_function(coalitions=coalitions)
        self.empty_value = float(game_output[0])
        self.full_value = float(game_output[1])

        if verbose:
            print(f"Similarly of the Image and Text: {self.full_value} (empty_value={self.empty_value})")

        super().__init__(
            n_players=self.n_players_image + self.n_players_text,
            normalize=True,
            normalization_value=self.empty_value,
        )


    def _processor_function(self, input_image, input_text):
        """
        Input: list of images of length N, list of texts of length M.
        Output: a dictionary of processed inputs with {'input_ids', 'attention_mask', 'pixel_values'}
        """
        if self.model_type == "siglip" or self.model_type == "siglip2": 
            inputs = self.processor(
                images=input_image, 
                text=input_text, 
                return_tensors="pt", 
                padding="max_length",
                max_length=64
            )
        elif self.model_type == "clip": 
            inputs = self.processor(
                images=input_image, 
                text=input_text, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            )
        return inputs


    def value_function(self, coalitions, batch_size=None):
        """ Baseline value function
        Input: Coalitions of the game as a boolean np.array of shape (n_coalitions, n_players).
        Output: Model outputs for the coalitions of shape (n_coalitions, )."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions = coalitions.shape[0]
        coalitions_image = torch.from_numpy(coalitions[:, :self.n_players_image])
        coalitions_text = torch.from_numpy(coalitions[:, self.n_players_image:])

        if self.model_type == "siglip2": 
            # [n_coallitions, 64]
            text_attention_masks = torch.cat(
                (coalitions_text, torch.ones(n_coalitions, 64 - self.n_players_text)), 
                axis=1
            ).int()
        elif self.model_type == "siglip":
            # [n_coallitions, 64]
            text_attention_masks = torch.cat(
                (coalitions_text, torch.ones(n_coalitions, 64 - self.n_players_text)), 
                axis=1
            ).int()
        elif self.model_type == "clip": 
            # [n_coallitions, n_players_text + 2]
            text_attention_masks = torch.cat(
                (torch.ones(n_coalitions, 1), coalitions_text, torch.ones(n_coalitions, 1)), 
                axis=1
            ).int()
        # [n_coallitions, n_channels, image_size, image_size]
        image_binary_masks = self._generate_image_binary_mask(coalitions_image)
        # {'input_ids', 'attention_mask', 'pixel_values'}
        inputs_original = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_size)

        #:# batch processing
        batch_iters = n_coalitions // batch_size
        batch_left = n_coalitions % batch_size
        coalitions_outputs = []
        for batch_index in range(batch_iters + 1):
            if batch_index < batch_iters:
                inputs = {
                    "input_ids": inputs_original["input_ids"],
                    "attention_mask": text_attention_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)],
                    "pixel_values": inputs_original["pixel_values"] * image_binary_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)]
                }
            elif batch_left > 0: # process last batch (once)
                inputs_left = self._processor_function([self.input_image]*batch_left, [self.input_text]*batch_left)
                inputs = {
                    "input_ids": inputs_left["input_ids"],
                    "attention_mask": text_attention_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)],
                    "pixel_values": inputs_left["pixel_values"] * image_binary_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)]
                }
            else:
                break 
            with torch.no_grad():
                inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
                if self._force_float32:
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float32)
                outputs = self.model(**inputs)
            # take only the diagonal predictions - a naive approach
            outputs = torch.diagonal(outputs.logits_per_image).cpu()
            coalitions_outputs.append(outputs)
        coalitions_outputs = torch.concat(coalitions_outputs)

        return coalitions_outputs.numpy()


    def value_function_crossmodal(self, coalitions_image, coalitions_text, batch_size=None):
        """ Efficient value function
        Input: Coalitions of the game as two boolean np.arrays of shapes 
            (n_coalitions_image, n_players_image) and (n_coalitions_text, n_players_text).
        Output: Model outputs for the coalitions of shape (n_coalitions_image, n_coalitions_text)."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions_image = coalitions_image.shape[0]
        n_coalitions_text = coalitions_text.shape[0]

        if self.model_type == "siglip2":
            # [n_coallitions, 64]
            text_attention_masks = torch.cat(
                (torch.from_numpy(coalitions_text), torch.ones(n_coalitions_text, 64 - self.n_players_text)), 
                axis=1
            ).int()
        elif self.model_type == "siglip":
            # [n_coallitions, 64]
            text_attention_masks = torch.cat(
                (torch.from_numpy(coalitions_text), torch.ones(n_coalitions_text, 64 - self.n_players_text)), 
                axis=1
            ).int()
        elif self.model_type == "clip": 
            # [n_coallitions, n_players_text + 2]
            text_attention_masks = torch.cat(
                (torch.ones(n_coalitions_text, 1), torch.from_numpy(coalitions_text), torch.ones(n_coalitions_text, 1)), 
                axis=1
            ).int()

        # [n_coalitions_image, n_channels, image_size, image_size]
        image_binary_masks = self._generate_image_binary_mask(torch.from_numpy(coalitions_image))
        # {'input_ids', 'attention_mask', 'pixel_values'}
        inputs_original = self._processor_function([self.input_image]*batch_size, [self.input_text]*batch_size)

        #:# batch processing
        coalitions_outputs = []
        for img_start in range(0, n_coalitions_image, batch_size):
            img_end = min(img_start + batch_size, n_coalitions_image)
            n_img = img_end - img_start
            current_pixel_values = inputs_original['pixel_values'][:n_img] * image_binary_masks[img_start:img_end]
            coalitions_outputs_image = []
            for txt_start in range(0, n_coalitions_text, batch_size):
                txt_end = min(txt_start + batch_size, n_coalitions_text)
                n_txt = txt_end - txt_start
                inputs = {
                    "input_ids": inputs_original["input_ids"][:n_txt],
                    "pixel_values": current_pixel_values,
                    "attention_mask": text_attention_masks[txt_start:txt_end]
                }
                with torch.no_grad():
                    inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
                    if self._force_float32:
                        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float32)
                    outputs = self.model(**inputs)
                outputs = outputs.logits_per_image.cpu()
                coalitions_outputs_image.append(outputs)
            coalitions_outputs.append(torch.concat(coalitions_outputs_image, axis=1))
        coalitions_outputs = torch.concat(coalitions_outputs, axis=0)

        return coalitions_outputs.numpy()
    

    #:# ---------- utility functions ---------- #:#

    def _generate_image_binary_mask(self, coalitions):
        """
        Input: binary torch tensor
        Output: binary torch tensor
        """
        n_coalitions = coalitions.shape[0]
        # Expand each coalition value into a patch
        binary_masks = coalitions\
            .repeat_interleave(self.patch_size**2, dim=1)\
                .reshape(n_coalitions, self.grid_size, self.grid_size, self.patch_size, self.patch_size)
        # Rearrange to form the final batch of full-size images
        binary_masks = binary_masks\
            .permute(0, 1, 3, 2, 4)\
                .reshape(n_coalitions, self.image_size, self.image_size)
        # Add image channel dimension
        binary_masks = binary_masks\
            .repeat((self.n_channels, 1, 1, 1))\
                .permute(1, 0, 2, 3)
        return binary_masks