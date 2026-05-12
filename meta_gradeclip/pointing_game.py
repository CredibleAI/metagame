import argparse
import os
from pathlib import Path

import torch
import shapiq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoProcessor

import fixlip
from metagame import Metagame, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-only", action="store_true",
                    help="Skip GPU sweep + per-image aggregation and reproduce "
                         "Tables 2 & 4 from results/pointing_game_results.csv directly.")
    args = ap.parse_args()
    TABLE_ONLY = args.table_only

    PATH_OUTPUT = os.environ.get("POINTING_GAME_OUTPUT", "results")
    print(device)

    if not TABLE_ONLY:
        # Pointing-game model variants commented; switch by uncommenting the
        # corresponding (MODEL_NAME, n_image_tokens) pair.
        # MODEL_NAME = "openai/clip-vit-base-patch16"
        # n_image_tokens = 196
        # MODEL_NAME = "google/siglip2-base-patch32-256"
        # n_image_tokens = 64
        # MODEL_NAME = "google/siglip2-large-patch16-256"
        # n_image_tokens = 256
        MODEL_NAME = "facebook/metaclip-2-worldwide-huge-quickgelu"
        n_image_tokens = 256
        model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.config.vision_config._attn_implementation = "eager"
        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        GAMES = [
            ['fish', 'koala', 'balloon', 'laptop'],
            ['koala', 'plane', 'church', 'lemon'],
            ['plane', 'balloon', 'jeep', 'pizza'],
            ['balloon', 'church', 'laptop', 'acorn'],
            ['church', 'jeep', 'lemon', 'fish'],
            ['jeep', 'laptop', 'pizza', 'koala'],
            ['laptop', 'lemon', 'acorn', 'plane'],
            ['lemon', 'pizza', 'fish', 'balloon'],
            ['pizza', 'acorn', 'koala', 'church'],
            ['acorn', 'fish', 'plane', 'jeep']
        ]

        def denormalize(img, processor):
            return img * torch.tensor(processor.image_std).view(3, 1, 1) + torch.tensor(processor.image_mean).view(3, 1, 1)

        DATASET_ROOT = os.environ.get("POINTING_GAME_DATASET", "pointing_game")
        for mode in ['gradeclip', 'genericattention', 'maskclip']:
        # for mode in ['gradesiglip', 'genericattention', 'masksiglip']:
            for game in GAMES:
                PATH_INPUT = os.path.join(DATASET_ROOT, "_".join(game))
                for i_objects in range(1, 5):
                    class_labels = game[:i_objects]
                    cl = "_".join(class_labels)
                    input_text = cl.replace("_", " ")
                    path_output = os.path.join(PATH_OUTPUT, MODEL_NAME, mode, cl)
                    if not os.path.exists(path_output):
                        os.makedirs(path_output)
                    for i_image in range(50):
                        input_image = Image.open(os.path.join(PATH_INPUT, f'{i_image}.jpg'))
                        meta_explainer = Metagame(model, processor, n_image_tokens=n_image_tokens, mode=mode)
                        meta_values = meta_explainer.explain(input_image, input_text)
                        meta_values.save(Path(os.path.join(path_output, f'meta_{i_image}')))

                        values = meta_explainer.first_order_explanations[tuple(range(i_objects))].reshape(-1)
                        values = {(): 0} | {(i,): val.item() for i, val in enumerate(values)} | {(i,): 0 for i in range(n_image_tokens, n_image_tokens + i_objects)}
                        attribution_values = shapiq.InteractionValues(
                            values=values,
                            index=mode,
                            n_players=n_image_tokens + i_objects,
                            max_order=1,
                            min_order=1,
                            baseline_value=0
                        )
                        attribution_values.save(Path(os.path.join(path_output, f'{i_image}')))

                        image_preprocessed = processor(input_image, return_tensors="pt")["pixel_values"][0]
                        input_image_denormalized = denormalize(image_preprocessed, processor.image_processor)
                        input_image_denormalized = input_image_denormalized.permute(1, 2, 0).numpy()
                        fig = fixlip.plot.plot_image_and_text_together(
                            img=input_image_denormalized,
                            text=cl.split("_"),
                            image_players=list(range(n_image_tokens)),
                            iv=meta_values,
                            plot_interactions=True,
                            top_k=30,
                            normalize_jointly=True,
                            figsize=(5, 5),
                            fontsize=22,
                            margin=0.3,
                            color_text=False,
                            plot_heatmap=False,
                            show=False
                        )
                        fig.suptitle(f'{MODEL_NAME} {mode}', fontsize=20, y=1.05)
                        fig.savefig(os.path.join(path_output, f'meta_{i_image}.png'), bbox_inches='tight')
                        plt.close(fig)

        # Measure pointing-game recognition

        grid_players = {
            8: {
                0: np.tile(range(0, 4), 4) + np.repeat(np.arange(0, 4) * 8, 4),
                1: np.tile(range(4, 8), 4) + np.repeat(np.arange(0, 4) * 8, 4),
                2: np.tile(range(0, 4), 4) + np.repeat(np.arange(0, 4) * 8, 4) + 8 * 4,
                3: np.tile(range(4, 8), 4) + np.repeat(np.arange(0, 4) * 8, 4) + 8 * 4
            },
            14: {
                0: np.tile(range(0, 7),  7) + np.repeat(np.arange(7) * 14, 7),
                1: np.tile(range(7, 14), 7) + np.repeat(np.arange(7) * 14, 7),
                2: np.tile(range(0, 7),  7) + np.repeat(np.arange(7) * 14, 7) + 14 * 7,
                3: np.tile(range(7, 14), 7) + np.repeat(np.arange(7) * 14, 7) + 14 * 7
            },
            16: {
                0: np.tile(range(0, 8),  8) + np.repeat(np.arange(8) * 16, 8),
                1: np.tile(range(8, 16), 8) + np.repeat(np.arange(8) * 16, 8),
                2: np.tile(range(0, 8),  8) + np.repeat(np.arange(8) * 16, 8) + 16 * 8,
                3: np.tile(range(8, 16), 8) + np.repeat(np.arange(8) * 16, 8) + 16 * 8
            }
        }

        results = pd.DataFrame({
            'text_input': [],
            'n_objects': [],
            'image_id': [],
            'mass_ratio': [],
        })
        for model_name, grid_size in {
            "openai/clip-vit-base-patch16": 14, # 224
            "google/siglip2-base-patch32-256": 8,
            "google/siglip2-large-patch16-256": 16,
            "facebook/metaclip-2-worldwide-huge-quickgelu": 16
        }.items():
            if model_name.startswith("openai/clip"):
                modes = ['gradeclip', 'genericattention', 'maskclip', 'fixlip']
            elif model_name.startswith("google/siglip2"):
                modes = ['gradesiglip', 'genericattention', 'masksiglip', 'fixlip']
            elif model_name.startswith("facebook/metaclip"):
                modes = ['gradeclip', 'genericattention', "fixlip"]
            for mode in modes:
                if mode == "fixlip":
                    metas = [False]
                else:
                    metas = [False, True]
                for input_text in os.listdir(os.path.join(PATH_OUTPUT, model_name, mode)):
                    class_labels = input_text.split("_")
                    n_objects = len(class_labels)
                    for image_id in range(50):
                        for meta in metas:
                            path_file = os.path.join(PATH_OUTPUT, model_name, mode, input_text, f'{"meta_" if meta else ""}{image_id}.json')
                            try:
                                iv = shapiq.InteractionValues.load(path_file)
                            except:
                                continue
                            n_players_image = iv.n_players - n_objects
                            grid_ids = grid_players[grid_size]
                            mass_correct, mass_wrong = 0, 0

                            iv_subset = fixlip.utils.get_crossmodal_subset(iv, n_players_image, n_objects)

                            for token_id, token_text in enumerate(class_labels):
                                image_players_in = grid_ids[token_id]
                                text_players_out = n_players_image + np.array([e for e in range(len(class_labels)) if e != token_id])
                                image_players_out = np.concat([grid_ids[e] for e in range(4) if e != token_id])
                                if meta or mode == "fixlip":
                                    iv_subset_in = fixlip.utils.get_subset(iv_subset, players=np.append(image_players_in, n_players_image + token_id), rename_players=False)
                                    iv_subset_out = fixlip.utils.get_subset(iv_subset, players=np.append(image_players_out, n_players_image + token_id), rename_players=False)
                                    values_in = iv_subset_in.get_n_order(2).values
                                    values_out = iv_subset_out.get_n_order(2).values
                                else:
                                    iv_subset_in = fixlip.utils.get_subset(iv_subset, players=np.append(image_players_in, n_players_image + token_id), rename_players=False)
                                    iv_subset_out = fixlip.utils.get_subset(iv_subset, players=np.append(image_players_out, n_players_image + token_id), rename_players=False)
                                    values_in = iv_subset_in.get_n_order(1).values
                                    values_out = iv_subset_out.get_n_order(1).values

                                mass_correct += values_in[values_in > 0].sum().item() + np.abs(values_out[values_out < 0]).sum().item()
                                mass_wrong += values_out[values_out > 0].sum().item() + np.abs(values_in[values_in < 0]).sum().item()

                            results = pd.concat([results, pd.DataFrame({
                                'model_name': [model_name],
                                'mode': [mode],
                                'meta': [meta],
                                'n_objects': [n_objects],
                                'text_input': [" ".join(class_labels)],
                                'image_id': [image_id],
                                'mass_ratio': [mass_correct / (mass_correct + mass_wrong)],
                            })])

        results.to_csv(os.path.join(PATH_OUTPUT, "pointing_game_results.csv"), index=False)

    results = pd.read_csv(os.path.join(PATH_OUTPUT, "pointing_game_results.csv"))
    pd.set_option('display.max_rows', 120)
    print(results.groupby(["model_name", "mode", "meta", "n_objects"]).agg(
        mean=('mass_ratio', 'mean'),
        se2=('mass_ratio', lambda x: 2 * x.std() / np.sqrt(x.count())),
    ).round(3))


if __name__ == "__main__":
    main()
