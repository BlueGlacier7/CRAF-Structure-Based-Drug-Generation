import argparse
import time
from scripts import CRAF_main, Generate
from scripts.utils import *
from scripts.utils import mask_node, Protein, ComplexData, ComplexData
import collections
import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key in config:
        if isinstance(config[key], str):
            if config[key].lower() in {'true', 'false'}:
                config[key] = str2bool(config[key])
            elif config[key].replace('.', '', 1).isdigit():
                if '.' in config[key]:
                    config[key] = float(config[key])
                else:
                    config[key] = int(config[key])
    return argparse.Namespace(**config)

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args = load_config()
    if args.name == 'receptor':
        args.name = args.pocket.split('/')[-1].split('-')[0]
    ## Load Target
    assert args.pocket != 'None', 'Please specify pocket !'
    assert args.ckpt != 'None', 'Please specify model !'
    pdb_file = args.pocket
    args.choose_max = str2bool(args.choose_max)
    args.with_print = str2bool(args.with_print)

    pro_dict = Protein(pdb_file).get_atom_dict(removeHs=True, get_surf=True)
    lig_dict = Ligand.empty_dict()
    data = ComplexData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pro_dict),
                    ligand_dict=torchify_dict(lig_dict),
                )

    ## init transform
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
    focal_masker = FocalMaker(r=6.0, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
    atom_composer = AtomComposer(knn=16, num_workers=16, for_gen=True, use_protein_bond=True)

    ## transform data
    data = RefineData()(data)
    data = LigandCountNeighbors()(data)
    data = protein_featurizer(data)
    data = ligand_featurizer(data)
    node4mask = torch.arange(data.ligand_pos.size(0))
    data = mask_node(data, torch.empty([0], dtype=torch.long), node4mask, num_atom_type=9, y_pos_std=0.)
    data = atom_composer.run(data)

    ## Load model
    print('Loading model ...')
    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)

    config = ckpt['config']
    model = CRAF_main(config).to(device)
    model.load_state_dict(ckpt['model'])
    print('Generating molecules ...')
    temperature = [args.atom_temperature, args.bond_temperature]
    if isinstance(args.bond_length_range, str):
        args.bond_length_range = eval(args.bond_length_range)
    generate = Generate(model, atom_composer.run, temperature=temperature, atom_type_map=[6,7,8,9,15,16,17,35,53],
                        num_bond_type=4, max_atom_num=args.max_atom_num, focus_threshold=args.focus_threshold, 
                        max_double_in_6ring=args.max_double_in_6ring, min_dist_inter_mol=args.min_dist_inter_mol,
                        bond_length_range=args.bond_length_range, choose_max=args.choose_max, device=device)
    start = time.time()
    generate.generate(data, num_gen=args.num_gen, rec_name=args.name, with_print=args.with_print,
                      root_path=args.root_path)
    os.system('cp {} {}'.format(args.ckpt, generate.out_dir))
    
    gen_config = '\n'.join(['{}: {}'.format(k,v) for k,v in args.__dict__.items()])
    with open(generate.out_dir + '/readme.txt', 'w') as fw:
        fw.write(gen_config)
    end = time.time()
    print('Time: {}'.format(timewait(end-start)))

