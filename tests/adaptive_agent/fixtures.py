"""Small public-input fixtures shared by the tracked adaptive regression suite."""
from PIL import Image


def episode(root, ds):
    Image.new('RGB', (100, 100), 'gray').save(root/'map.png')
    return dict(episode_id='fixture', agent_input=dict(dataset=ds,
        setting='visual' if ds == 'AVDN' else 'landmark_assisted',
        instruction_raw='Find the red car beside the house', turns=[],
        initial_pose=dict(xy_m=[50., 50.], heading_deg=0.),
        coordinate_frame='east/north meters', named_landmarks=[], views=[]),
        initialization=dict(direction=0., final_target='red car beside the house', auxiliary=[]),
        environment=dict(image_path=str(root/'map.png'), image_size=[100, 100],
            world_to_raster=[[1, 0, 0], [0, -1, 100], [0, 0, 1]],
            main_width_m=40., main_height_m=40.))
