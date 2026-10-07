import open3d as o3d
from ingestion.sequence import SequenceLoader

seq = SequenceLoader(data_root="data/raw", split="training", sequence="0001")
frame = seq[0]

print(frame)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(frame.xyz)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

opt = vis.get_render_option()
opt.background_color = [0, 0, 0]  

vis.run()
vis.destroy_window()