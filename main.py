from utils.video_utils import read_video, save_video
from trackers import Trackers
import cv2
from team_assigner import TeamAssigner

def main():
    video_frames = read_video('Input\\football_video.mp4')

    tracker = Trackers('models/best.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/tracks_stubs.pkl'
    )

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
