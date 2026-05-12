import argparse
parser = argparse.ArgumentParser(description='FIxLIP attribution sweep for one (model, scene).')
parser.add_argument('--model_name', type=str)
parser.add_argument('--path_input', type=str)
parser.add_argument('--path_output', type=str)
parser.add_argument('--class_labels', type=str)
parser.add_argument('--budget', type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--random_state', default=0, type=int)
args = parser.parse_args()
MODEL_NAME = args.model_name
PATH_INPUT = args.path_input
PATH_OUTPUT = args.path_output
CLASS_LABELS = args.class_labels
BUDGET = args.budget
BATCH_SIZE = args.batch_size
RANDOM_STATE = args.random_state

print(f'-- Input: {PATH_INPUT}', flush=True)
print(f'-- Output: {PATH_OUTPUT}', flush=True)

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'- Device: {DEVICE}', flush=True)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
    torch.backends.cudnn.enabled = False
from transformers import AutoModel, AutoProcessor

from pathlib import Path
import os
if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

import fixlip  # noqa: E402
fixlip.utils.set_seed(RANDOM_STATE)

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
else:
    model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
input_text = CLASS_LABELS.replace("_", " ")

for i in tqdm(range(50)):
    input_image = Image.open(os.path.join(PATH_INPUT, f'{i}.jpg'))
    game = fixlip.game_huggingface.VisionLanguageGame(
        model, processor,
        input_image=input_image,
        input_text=input_text,
        batch_size=BATCH_SIZE
    )
    explainer = fixlip.fixlip.FIxLIP(
        n_players=game.n_players,
        mode="shapley",
        max_order=2,
        random_state=RANDOM_STATE
    )

    if game.n_players_image == 64:
        top_k = 5 * game.n_players_text
        interaction_lookup = None
    elif game.n_players_image == 196 or game.n_players_image == 256:
        top_k = 20 * game.n_players_text
        interaction_lookup = fixlip.utils.create_crossmodal_interaction_lookup(game.n_players_image, game.n_players_text)

    interaction_values = explainer.approximate(
        game=game,
        budget=BUDGET,
        interaction_lookup=interaction_lookup
    )
    interaction_values.save(Path(os.path.join(PATH_OUTPUT, f'{i}.json')))

    ## visualize explanations
    text_tokens = CLASS_LABELS.split("_")
    assert len(text_tokens) == game.n_players_text
    players_text = list(range(game.n_players_image, game.n_players))
    assert game.n_players == interaction_values.n_players == max(players_text) + 1
    input_image_processed = game.inputs['pixel_values'].squeeze(0)
    input_image_denormalized = fixlip.utils.denormalize(
        input_image_processed,
        game.processor.image_processor.image_mean,
        game.processor.image_processor.image_std
    ).permute(1, 2, 0).numpy()
    fig = fixlip.plot.plot_image_and_text_together(
        img=input_image_denormalized,
        text=text_tokens,
        image_players=list(range(game.n_players_image)),
        iv=interaction_values,
        plot_interactions=True,
        top_k=top_k,
        normalize_jointly=True,
        figsize=(5, 5),
        fontsize=22,
        margin=0.3,
        color_text=False,
        plot_heatmap=False,
        show=False
    )
    fig.suptitle(f'{MODEL_NAME} fixlip', fontsize=20, y=1.05)
    fig.savefig(os.path.join(PATH_OUTPUT, f'{i}.png'), bbox_inches='tight')
    plt.close(fig)