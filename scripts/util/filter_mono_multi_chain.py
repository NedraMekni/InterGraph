import os 

from functools import reduce

in_dir = './out_conect_def'
out_dir_multichain = './out_conect_multi_def'
out_dir_monochain = './out_conect_mono_def'


def write_multi_mono_dir():
    all_conect_pdb = os.listdir(in_dir)
    all_conect_pdb_protein = [x.split('_')[0] for x in all_conect_pdb]
    all_conect_pdb_protein = {k:[x for x in all_conect_pdb if x.split('_')[0]==k] for k in set(all_conect_pdb_protein)}
    mono_dir_fnames = [all_conect_pdb_protein[k][0] for k in all_conect_pdb_protein.keys() if len(all_conect_pdb_protein[k])==1 ] 
    multi_dir_fnames = [all_conect_pdb_protein[k] for k in all_conect_pdb_protein.keys() if len(all_conect_pdb_protein[k])>1 ] 

    multi_dir_fnames = reduce(lambda x,y:x+y,multi_dir_fnames)
    assert len([x for x in multi_dir_fnames if x in mono_dir_fnames])==0
    print('monodir_len = {}, multidir_len = {}'.format(len(mono_dir_fnames),len(multi_dir_fnames)))

    print('... copy mono in {}'.format(out_dir_monochain))
    for mono_dir_fname in mono_dir_fnames:
        os.system('cp '+in_dir+'/'+mono_dir_fname+' '+out_dir_monochain+'/'+mono_dir_fname)
    print('... copy multi in {}'.format(out_dir_multichain))
    for multi_dir_fname in multi_dir_fnames:
        os.system('cp '+in_dir+'/'+multi_dir_fname+' '+out_dir_multichain+'/'+multi_dir_fname)

if __name__=='__main__':
    write_multi_mono_dir()