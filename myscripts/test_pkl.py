import pickle

with open('data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl', 'rb') as f:
    infos = pickle.load(f)

print('metadata=', infos['metadata'])
print(f"len(infos['infos'])={len(infos['infos'])}")
print("infos['infos'][0]=")
for key in infos['infos'][0]:
    if key == 'next': continue
    print(key, "=", infos['infos'][0][key])
# import pdb
# pdb.set_trace()

'''
pkl结构如下：
{
    'infos':[{'lidar_path':xxx, 'token':xxx, 'sweeps':xxx, 'gt_boxes':xxx,
              'gt_names':xxx,'gt_velocity':xxx, }, {}, {}...],
    'metadata':{'version': 'v1.0-trainval'}
}
'''
