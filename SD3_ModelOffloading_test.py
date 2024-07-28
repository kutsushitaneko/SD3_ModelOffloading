# pip install transformers, diffusers, accelerate, sentencepiece, protobuf, ipywidgets, pypng

import torch
from diffusers import StableDiffusion3Pipeline
import os
import png
import time


def create_pipe(offload = True):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
    )
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def inference(pipe, seed):
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt = prompt,
        prompt_3=prompt_3,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=generator
        ).images[0]
    
    return image


def make_text_chunk(keyword, text):
    # PNG のテキストチャンクを組み立てる
    return keyword.encode('latin-1') + b'\0' + text.encode('utf-8')

def save_image_with_metadata(image, base_filename, prompt=None, prompt_3=None, seed=1):
    # 出力ディレクトリを作成
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 連番のファイル名を生成
    i = 1
    while True:
        filename = os.path.join(output_dir, f"{base_filename}_{i:03d}.png")
        if not os.path.exists(filename):
            break
        i += 1

    # 画像を保存
    image.save(filename)
    
    # テキストチャンクを作成
    text_chunks = []
    if prompt:
        text_chunks.append((b'tEXt', make_text_chunk('prompt', prompt)))
    if prompt_3:
        text_chunks.append((b'tEXt', make_text_chunk('prompt_3', prompt_3)))
    if seed:
        text_chunks.append((b'tEXt', make_text_chunk('seed', str(seed))))
    
    if text_chunks:
        # PNGファイルを読み込み、テキストチャンクを追加して再保存
        reader = png.Reader(filename=filename)
        chunks = reader.chunks()
        chunk_list = list(chunks)
        
        # IHDRチャンクの直後にテキストチャンクを挿入
        insert_pos = 1
        for chunk in text_chunks:
            chunk_list.insert(insert_pos, chunk)
            insert_pos += 1
        
        with open(filename, 'wb') as output:
            png.write_chunks(output, chunk_list)

if __name__ == "__main__":
    # 繰り返し回数を指定
    n = 1
    # モデルオフロードを有効にするかどうか
    offload = True
    # シード値のベース
    seed_base = 123456789012

    prompt = "Photorealistic graphics of a girl with long blue hair stands in front of a spaceship window, gazing outside. The scene captures her back view, emphasizing her contemplative posture as she admires the vastness of space."
    prompt_3 = "She is wearing a large white hat and a white dress. Outside the window, a vibrant galaxy is visible, filled with colorful stars and cosmic phenomena. The scene captures her back view, emphasizing her contemplative posture as she admires the vastness of space. The interior of the spaceship is simply constructed in metallic colors, and her figure floating in the dimness is impressive. The walls of the ship have a metallic sheen and glow dully in the dim light."

    # パイプラインを作成
    pipe = create_pipe(offload=offload)

    total_time = 0
    if offload:
        print("モデルオフロードが有効です")
    else:
        print("モデルオフロードが無効です")
    
    for i in range(n):
        start_time = time.time()
        image = inference(pipe, seed_base+i+1)
        end_time = time.time()
    
        inference_time = end_time - start_time
        total_time += inference_time
    
        save_image_with_metadata(image, "sd3_generated", prompt=prompt, prompt_3=prompt_3, seed=seed_base+i+1)
        print(f"{n}枚中 {i+1}枚目の画像が生成され、保存されました")
        print(f"推論時間: {inference_time:.2f}秒")

    average_time = total_time / n
    print(f"\n平均推論時間: {average_time:.2f}秒")