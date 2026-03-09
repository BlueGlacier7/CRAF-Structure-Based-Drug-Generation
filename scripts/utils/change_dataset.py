import io
import os
import sys
import lmdb
import pickle
import zlib
from tqdm import tqdm

# ...existing code...
def rebuild_lmdb(src_path, dst_path, old_prefix='pocket_flow', new_prefix='scripts', map_size=50*(1024**3), commit_interval=1000):
    class RenamingUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(old_prefix):
                module = module.replace(old_prefix, new_prefix, 1)
            return super().find_class(module, name)

    env_src = lmdb.open(src_path, readonly=True, lock=False, subdir=False)
    if os.path.exists(dst_path):
        raise FileExistsError(f"{dst_path} already exists")
    env_dst = lmdb.open(dst_path, map_size=map_size, create=True, subdir=False)

    keys = []
    with env_src.begin() as txn:
        keys = [k for k, _ in txn.cursor()]

    written = 0
    txn = env_dst.begin(write=True)
    try:
        for k in tqdm(keys, desc="Rebuilding lmdb"):
            v = env_src.begin().get(k)
            obj = None
            # try direct unpickle
            try:
                obj = RenamingUnpickler(io.BytesIO(v)).load()
            except Exception:
                # try zlib decompress then unpickle
                try:
                    data = zlib.decompress(v)
                    obj = RenamingUnpickler(io.BytesIO(data)).load()
                except Exception:
                    # fallback to normal loads (no renaming)
                    try:
                        obj = pickle.loads(v)
                    except Exception:
                        try:
                            obj = pickle.loads(zlib.decompress(v))
                        except Exception:
                            raise RuntimeError(f"Cannot unpickle key {k!r}")
            # re-dump with current environment (no old module refs)
            new_v = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(k, new_v)
            written += 1
            if written % commit_interval == 0:
                txn.commit()
                txn = env_dst.begin(write=True)
        txn.commit()
    finally:
        env_src.close()
        env_dst.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--old_prefix", default="pocket_flow")
    p.add_argument("--new_prefix", default="scripts")
    args = p.parse_args()
    rebuild_lmdb(args.src, args.dst, old_prefix=args.old_prefix, new_prefix=args.new_prefix)
# ...existing code...