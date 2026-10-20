#!/usr/bin/env python3
"""Pack Mirai's Qwen3.8-Flash-Next-S (QNS) GGUF once and write a server config for it (docs/QNS.md).

    .venv/bin/python tools/qns_config.py --model-dir DIR --ngram FILE [--size S|S-small] [--gpu 0[,1]]
                                          [--context 131072] [--vision on|off] [--vector on|off] [--port 8080]
    .venv/bin/python -m serve.server --engine strata --config strata-qns-S.json

DIR is the downloaded model folder: S/ or S-small/ with the two GGUFs, mtp/rt, the mmproj and the refusal vector
(the layout of the Hugging Face repo in docs/QNS.md). FILE is the n-gram table: shard 2 of any ISTA-DASLab size.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model-dir', required=True, type=Path)
ap.add_argument('--ngram', required=True, type=Path, help='the n-gram (PLE) table: ISTA-DASLab shard 2')
ap.add_argument('--size', default='S', choices=['S', 'S-small'])
ap.add_argument('--gpu', default='0', help='one card, or two or more for a layer split: 0,1')
ap.add_argument('--context', type=int, default=131072)
ap.add_argument('--vision', default='on', choices=['on', 'off'])
ap.add_argument('--vector', default='on', choices=['on', 'off'], help='the refusal projection (Huihui direction)')
ap.add_argument('--port', type=int, default=8080)
ap.add_argument('--out', type=Path)
a = ap.parse_args()

d = a.model_dir.resolve()
gguf = d / a.size / f'Qwen3.8-Flash-Next-{a.size}-qns.gguf'
dense = d / a.size / f'Qwen3.8-Flash-Next-{a.size}-qns-dense.gguf'
mtp, mmproj = d / 'mtp' / 'rt', d / 'mmproj-Qwen3.8-Flash-Next-BF16.gguf'
vector = d / 'Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf'
need = [gguf, dense, mtp / 'experts.bin', a.ngram] + ([mmproj] if a.vision == 'on' else []) + \
       ([vector] if a.vector == 'on' else [])
missing = [str(p) for p in need if not p.exists()]
if missing:
    sys.exit('missing: ' + ', '.join(missing))
engine, vis = ROOT / 'build' / 'strata', ROOT / 'build-vision' / 'bin' / 'strata-vision'
if not engine.exists():
    sys.exit(f'no engine at {engine}: run tools/build_qns_linux.sh first')

pack = d / a.size / 'pack'
if not (pack / 'native_experts.txt').exists():
    gguf_py = ROOT / 'build' / '_deps' / 'strata_llamacpp-src' / 'gguf-py'
    env = dict(os.environ, STRATA_GGUF_PY=str(gguf_py), CUDA_VISIBLE_DEVICES='')
    print(f'packing {gguf.name} -> {pack} (seconds)')
    subprocess.run([sys.executable, str(ROOT / 'tools' / 'iq_pack.py'), '--gguf', str(gguf), '--out', str(pack)],
                   check=True, env=env)

gpus = [int(g) for g in a.gpu.split(',')]
args = ['--pack', str(pack), '--native', str(gguf), '--native-dense-gguf', str(gguf), '--qns-dense-gguf', str(dense),
        '--ple-gguf', str(a.ngram.resolve()), '--expert-profile', str(ROOT / 'data' / 'expert-profile.bin'),
        '--expert-cache', 'auto', '--prefill', 'auto', '--spec', '4', '--spec-min-p', '0.5', '--mtp', str(mtp),
        '--max-context', str(a.context), '--kv', 'int8', '--vram-reserve-mib', '700']
if a.vector == 'on':
    args += ['--control-vector-scaled', f'{vector}:1.0', '--control-vector-layer-range', '1', '47',
             '--cvec-mode', 'project', '--cvec-dir', 'per-layer']
cfg = {'exe': str(engine), 'args': args, 'cwd': str(ROOT), 'tokenizer': str(pack / 'tokenizer'),
       'model_name': f'qwen3.8-flash-next-{a.size.lower()}', 'log': str(ROOT / f'strata-qns-{a.size}.log'),
       'host': '127.0.0.1', 'port': a.port, 'gpu': gpus, 'fit_max_tokens': True,
       'sampling': {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95, 'min_p': 0.0, 'presence_penalty': 0.0,
                    'repetition_penalty': 1.0}}
if len(gpus) > 1:
    cfg['layer_split'] = 'auto'
if a.vision == 'on':
    if not vis.exists():
        sys.exit(f'no picture reader at {vis}: run tools/build_qns_linux.sh, or pass --vision off')
    args.insert(0, '--vision')
    cfg['vision'] = {'exe': str(vis), 'mmproj': str(mmproj), 'model': str(gguf), 'gpu': True, 'max_tokens': 1024}
out = a.out or ROOT / f'strata-qns-{a.size}.json'
out.write_text(json.dumps(cfg, indent=1) + '\n')
print(f'wrote {out}\nstart: .venv/bin/python -m serve.server --engine strata --config {out}')
