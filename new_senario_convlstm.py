#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Config:
    # エリア設定
    AREA_WIDTH: float = 25.0
    AREA_HEIGHT: float = 10.0
    GRID_WIDTH: int = 50
    GRID_HEIGHT: int = 20
    
    # センサー設定
    SENSOR_POSITIONS: np.ndarray = None
    SENSOR_DETECTION_RANGE: float = 5.0
    
    # 歩行者設定
    MAX_PEOPLE: int = 3
    MIN_SPEED: float = 0.5
    MAX_SPEED: float = 2.5
    MIN_Y_POSITION: float = 2.5
    MAX_Y_POSITION: float = 7.5
    
    def __post_init__(self):
        if self.SENSOR_POSITIONS is None:
            self.SENSOR_POSITIONS = np.array([
                [7.5, 5.0],   # 左側
                [12.5, 5.0],  # 中央
                [17.5, 5.0]   # 右側
            ])
        self.MAX_Y_POSITION = self.AREA_HEIGHT - 2.5

# グローバル設定をインスタンス化
cfg = Config()

# 元コードとの互換性のためにグローバル変数も定義
AREA_WIDTH = cfg.AREA_WIDTH
AREA_HEIGHT = cfg.AREA_HEIGHT
GRID_WIDTH = cfg.GRID_WIDTH
GRID_HEIGHT = cfg.GRID_HEIGHT
SENSOR_POSITIONS = cfg.SENSOR_POSITIONS
SENSOR_DETECTION_RANGE = cfg.SENSOR_DETECTION_RANGE
MAX_PEOPLE = cfg.MAX_PEOPLE
MIN_SPEED = cfg.MIN_SPEED
MAX_SPEED = cfg.MAX_SPEED
MIN_Y_POSITION = cfg.MIN_Y_POSITION
MAX_Y_POSITION = cfg.MAX_Y_POSITION


def setup_and_visualize_sensors():
    # センサー配置の可視化
    plt.figure(figsize=(15, 6))  # 元のアスペクト比を保持
    plt.scatter(SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1], c='blue', marker='o', label='Sensors')
    
    # センサーの検知範囲を表示
    for i, pos in enumerate(SENSOR_POSITIONS):
        # 検知範囲の円を描画
        circle = plt.Circle((pos[0], pos[1]), SENSOR_DETECTION_RANGE, fill=False, color='green', alpha=0.3)
        plt.gca().add_artist(circle)
        
        # センサーラベルの追加
        offset_y = 0.3
        label = f'Sensor{i}'
        plt.annotate(label, 
                    (pos[0], pos[1] + offset_y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')
    
    # グリッド線の設定
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 補助線を追加
    plt.xticks(np.arange(0, AREA_WIDTH + 1, 5))
    plt.yticks(np.arange(0, AREA_HEIGHT + 1, 2))
    plt.xlabel('X (m)')
    plt.xlim(0, AREA_WIDTH)
    plt.ylabel('Y (m)')
    plt.ylim(0, AREA_HEIGHT)
    plt.title('Sensor Positions and Detection Ranges')
    plt.show()
    
    return SENSOR_POSITIONS

class PersonMotion:
    """歩行者の移動をシミュレートするクラス（元コード完全準拠）"""
    def __init__(self, init_x, init_y, direction, speed):
        self.x = init_x
        self.y = init_y
        self.direction = direction
        self.speed = speed
        
    def update_position(self, dt=1.0):
        """歩行者の位置を更新"""
        # 移動量の計算：速度×時間×方向(右:1, 左:-1)
        dx = self.speed * dt * (1 if self.direction == 'right' else -1)
        
        # x座標の更新
        self.x += dx
        
        return self.check_bounds()
    
    def check_bounds(self):
        """歩行者がエリア外に出たかどうかを確認"""
        if (self.direction == 'right' and self.x >= AREA_WIDTH) or (self.direction == 'left' and self.x <= 0):
            return True  # エリア外に出た
        return False     # まだエリア内にいる

def generate_heatmap(sensor_positions, people_positions, grid_size=(GRID_WIDTH, GRID_HEIGHT)):
    """ヒートマップを生成（元コード完全準拠）"""
    heatmap = np.zeros(grid_size)
    
    # センサーの検知状態を計算
    sensor_data = np.zeros(len(sensor_positions))  
    
    # センサーの位置と人の位置をマトリクスで距離計算
    for sensor_idx, sensor_pos in enumerate(sensor_positions):
        for x, y in people_positions:
            # センサーと人との距離を計算
            distance = np.sqrt((sensor_pos[0] - x)**2 + (sensor_pos[1] - y)**2)
            # 検知範囲以内に人がいればセンサーを1に設定
            if distance <= SENSOR_DETECTION_RANGE:
                sensor_data[sensor_idx] = 1
                break  # 1人でも検知したらその時点で次のセンサーへ
    
    # 検知しているセンサーごとにヒートマップを更新
    for sensor_idx, sensor_pos in enumerate(sensor_positions):
        if sensor_data[sensor_idx] == 1:  # センサーが検知している場合
            # 各グリッドセルについて
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    # グリッドセルの実際の座標を計算
                    grid_x = (i + 0.5) * (AREA_WIDTH / grid_size[0])
                    grid_y = (j + 0.5) * (AREA_HEIGHT / grid_size[1])
                    # センサーとグリッドセルの距離を計算
                    distance = np.sqrt((sensor_pos[0] - grid_x)**2 + (sensor_pos[1] - grid_y)**2)
                    # 検知範囲以内のグリッドセルの値を増加
                    if distance <= SENSOR_DETECTION_RANGE:
                        heatmap[i, j] += 1
    
    return heatmap, sensor_data

def visualize_heatmap_with_people(heatmap, sensor_positions, people_positions, sensor_data, grid_size=(GRID_WIDTH, GRID_HEIGHT), show_grid_values=True):
    plt.figure(figsize=(15, 5))  # 元のアスペクト比を保持
    plt.imshow(heatmap.T, origin='lower', extent=[0, AREA_WIDTH, 0, AREA_HEIGHT], aspect='auto', 
               vmin=0, vmax=4 if np.max(heatmap) > 0 else 1, cmap='RdPu_r')
    plt.colorbar(label='Number of Detecting Sensors')
    
    # グリッドの値を表示
    if show_grid_values:
        dx = AREA_WIDTH / grid_size[0]
        dy = AREA_HEIGHT / grid_size[1]
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if heatmap[i, j] > 0:  # 値が0より大きい場合のみ表示
                    # グリッドの実際の座標位置を計算
                    grid_x = i * dx + dx/2  # グリッドセルの中心X座標
                    grid_y = j * dy + dy/2  # グリッドセルの中心Y座標
                    plt.text(grid_x, grid_y, 
                            f'{int(heatmap[i, j])}',
                            color='white', 
                            ha='center', 
                            va='center',
                            fontsize=8)
    
    # センサー位置と検知範囲を追加
    for sensor_idx, sensor_pos in enumerate(sensor_positions):
        is_detecting = sensor_data[sensor_idx] == 1
        
        # センサー位置をプロット
        plt.scatter(sensor_pos[0], sensor_pos[1], c='black', marker='.', s=50)
        
        circle_color = 'red' if is_detecting else 'white'
        line_width = 2 if is_detecting else 0.5
        circle = plt.Circle((sensor_pos[0], sensor_pos[1]), SENSOR_DETECTION_RANGE, 
                          fill=False, color=circle_color, linestyle='--', 
                          linewidth=line_width, alpha=0.5)
        plt.gca().add_artist(circle)
    
    # 実際の人の位置をプロット
    if people_positions:
        x_positions = [p[0] for p in people_positions]
        y_positions = [p[1] for p in people_positions]
        plt.scatter(x_positions, y_positions, c='yellow', marker='*', s=150, label='People')
    
    plt.title('Heatmap with Actual Positions\n(Red circles: Detecting, White circles: Not detecting)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


# ConvLSTM実装
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# モデル定義（元コード完全準拠）
class EnhancedMultiPersonCoordinatePredictor(nn.Module):
    def __init__(self, max_people=26, input_channels=1):
        super().__init__()
        
        self.max_people = max_people
        self.input_channels = input_channels
        
        # ConvLSTM層 - 単一チャネル入力に対応
        self.convlstm = ConvLSTM(
            input_dim=input_channels,  # 入力チャネル数を1に設定
            hidden_dim=[64, 96, 128],
            kernel_size=(3, 3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=True
        )
        
        # 時間的アテンション機構
        self.temporal_attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # マルチスケール空間特徴抽出
        self.spatial_encoder_branches = nn.ModuleList([
            # スケール1: 細かい特徴
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # スケール2: 中間的特徴
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=5, stride=4, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # スケール3: 大域的特徴
            nn.Sequential(
                nn.AdaptiveAvgPool2d((13, 5)),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        ])
        
        # 特徴融合層
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # 空間的アテンション機構
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 座標予測と信頼度予測
        self.coordinate_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, max_people * 2)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_people),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 入力の形状チェックと変換
        if x.dim() == 4:  # [B, T, H, W] の場合
            # チャネル次元を追加: [B, T, H, W] -> [B, T, 1, H, W]
            x = x.unsqueeze(2)
        elif x.dim() == 5:  # [B, T, H, W, C] の場合
            # 形状変換: [B, T, H, W, C] -> [B, T, C, H, W]
            x = x.permute(0, 1, 4, 2, 3)
        
        # ConvLSTM処理
        output_list, _ = self.convlstm(x)
        last_layer_output = output_list[-1]
        
        # 時間的アテンションの適用
        time_steps = last_layer_output.size(1)
        temporal_features = []
        
        for t in range(time_steps):
            feat_t = last_layer_output[:, t]
            attention_t = self.temporal_attention(feat_t)
            weighted_feat_t = feat_t * attention_t
            temporal_features.append(weighted_feat_t)
        
        temporal_feature = sum(temporal_features)
        
        # マルチスケール空間特徴抽出
        multiscale_features = []
        for branch in self.spatial_encoder_branches:
            multiscale_features.append(branch(temporal_feature))
            
        # 13x5サイズに統一
        resized_features = []
        for feat in multiscale_features:
            if feat.size(2) != 13 or feat.size(3) != 5:
                feat = nn.functional.interpolate(feat, size=(13, 5), mode='bilinear', align_corners=False)
            resized_features.append(feat)
            
        # 特徴の連結
        concatenated_features = torch.cat(resized_features, dim=1)
        
        # 特徴融合
        fused_features = self.feature_fusion(concatenated_features)
        
        # 空間的アテンションを適用
        spatial_attention_map = self.spatial_attention(fused_features)
        attended_features = fused_features * spatial_attention_map
        
        # 座標と信頼度の予測
        coordinates = self.coordinate_head(attended_features)
        coordinates = coordinates.view(batch_size, self.max_people, 2)
        
        confidences = self.confidence_head(attended_features)
        
        # 座標を入力グリッドのスケールに合わせる
        coordinates[:, :, 0] = coordinates[:, :, 0] * 50
        coordinates[:, :, 1] = coordinates[:, :, 1] * 20
        
        return coordinates, confidences


class PedestrianTrackingDataset(Dataset):
    def __init__(self, num_samples=100, max_people=MAX_PEOPLE, relative_frequencies=None, visualize=False):
        # センサーの位置情報を取得
        self.sensor_positions = setup_and_visualize_sensors()
        # 最大人数を保存
        self.max_people = max_people
        # 可視化オプション
        self.visualize = visualize
        
        # 相対頻度が指定されていない場合のデフォルト値
        if relative_frequencies is None:
            relative_frequencies = [0.1, 0.7, 0.8, 1.0][:max_people + 1]
            
            # max_peopleの長さに合わせて調整
            if len(relative_frequencies) != max_people + 1:
                # 長さを調整（線形補間）
                new_frequencies = np.interp(
                    np.linspace(0, 1, max_people + 1),
                    np.linspace(0, 1, len(relative_frequencies)),
                    relative_frequencies
                )
                relative_frequencies = new_frequencies
        
        # 相対頻度を確率に正規化
        probabilities = np.array(relative_frequencies) / sum(relative_frequencies)
        
        # データ保存用のリスト
        self.heatmap_sequences = []  # 各サンプルのヒートマップシーケンス
        self.positions_sequences = []  # 各サンプルの人の位置シーケンス
        self.num_people_sequences = []  # 各サンプルの時間ごとの人数

        print(f"生成するシナリオベースの歩行者データセット: {num_samples}サンプル")
        
        # 指定されたサンプル数だけデータを生成
        for sample_idx in range(num_samples):
            if sample_idx % 10 == 0:
                print(f"シナリオ生成中... {sample_idx}/{num_samples}")
            
            # 確率分布に従って人数を選択
            # 0人は含めず、1人から最大人数までの範囲で選択する
            initial_people_count = np.random.choice(range(1, max_people + 1), p=probabilities[1:] / sum(probabilities[1:]))
            
            # 歩行者を初期化
            people = []
            for _ in range(initial_people_count):
                # x座標は0またはAREA_WIDTHのいずれか（エリアの端）
                init_x = np.random.choice([0, AREA_WIDTH])
                # y座標は設定された範囲内
                init_y = np.random.uniform(MIN_Y_POSITION, MAX_Y_POSITION)
                # 方向はx座標によって決まる
                direction = 'right' if init_x == 0 else 'left'
                # 速度は設定された範囲内
                speed = np.random.uniform(MIN_SPEED, MAX_SPEED)
                # 歩行者オブジェクトを作成して追加
                people.append(PersonMotion(init_x, init_y, direction, speed))
            
            # このサンプルのシーケンスデータを保存
            sample_heatmaps = []
            sample_positions = []
            sample_num_people = []
            
            # シミュレーション実行（全員がエリアから出るまで）
            timestep = 0
            while people and timestep < 1000:  # 最大1000ステップまで（無限ループ防止）
                # 現在の歩行者の位置を収集
                current_positions = [(person.x, person.y) for person in people]
                sample_positions.append(current_positions)
                sample_num_people.append(len(current_positions))
                
                # ヒートマップを生成
                heatmap, sensor_data = generate_heatmap(
                    self.sensor_positions, current_positions, grid_size=(GRID_WIDTH, GRID_HEIGHT)
                )
                sample_heatmaps.append(heatmap)
                
                # 可視化（オプション）
                if self.visualize and sample_idx == 0 and timestep % 10 == 0:
                    visualize_heatmap_with_people(
                        heatmap, self.sensor_positions, current_positions, 
                        sensor_data, grid_size=(GRID_WIDTH, GRID_HEIGHT)
                    )
                
                # 歩行者の位置更新と範囲外チェック
                people_to_remove = []
                for i, person in enumerate(people):
                    # 位置更新
                    if person.update_position():
                        # エリア外に出た場合、削除リストに追加
                        people_to_remove.append(i)
                
                # エリア外に出た歩行者を削除（後ろから削除して添字ずれを防ぐ）
                for idx in sorted(people_to_remove, reverse=True):
                    del people[idx]
                
                timestep += 1
            
            # シーケンスデータをデータセットに追加
            self.heatmap_sequences.append(np.array(sample_heatmaps))
            self.positions_sequences.append(sample_positions)
            self.num_people_sequences.append(sample_num_people)
        
        print(f"シナリオベースのデータセット生成完了: {num_samples}サンプル")
        
        # シナリオベースデータセットの統計情報を表示
        self.display_scenario_statistics()
            
    def __len__(self):
        return len(self.heatmap_sequences)
    
    def __getitem__(self, idx):
        """データセットからサンプルを取得"""
        heatmap_sequence = torch.FloatTensor(self.heatmap_sequences[idx])
        positions_sequence = self.positions_sequences[idx]
        num_people = self.num_people_sequences[idx]
        
        return {
            'heatmap_sequence': heatmap_sequence,
            'positions_sequence': positions_sequence,
            'num_people': num_people
        }
    
    def display_scenario_statistics(self):
        """シナリオベースデータセットの統計情報を表示"""
        # シーケンス長の分布
        sequence_lengths = [len(seq) for seq in self.heatmap_sequences]
        
        # 歩行者数の分布（各シーケンスの初期人数）
        initial_people_counts = [len(seq[0]) for seq in self.positions_sequences]
        
        # 統計の表示
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(sequence_lengths, bins=20, alpha=0.7)
        plt.title('Distribution of Scenario Lengths')
        plt.xlabel('Scenario Length (timesteps)')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(initial_people_counts, bins=range(self.max_people + 2), 
                alpha=0.7, align='left')
        plt.title('Distribution of Initial People Count')
        plt.xlabel('Number of People')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # 基本統計量を出力
        print(f"シナリオ長の統計: 最小={min(sequence_lengths)}, 最大={max(sequence_lengths)}, 平均={np.mean(sequence_lengths):.2f}")
        print(f"初期歩行者数の統計: 最小={min(initial_people_counts)}, 最大={max(initial_people_counts)}, 平均={np.mean(initial_people_counts):.2f}")
    
    def create_sliding_window_dataset(self, window_size=20, stride=5):
        """
        スライディングウィンドウ方式でデータセットを生成
        """
        windowed_heatmaps = []
        windowed_positions = []
        windowed_num_people = []
        scenario_indices = []  # 元のシナリオのインデックスを保存
        
        print(f"スライディングウィンドウ方式でデータセットを生成（ウィンドウサイズ={window_size}, ストライド={stride}）")
        
        total_windows = 0
        
        # 各シナリオに対して
        for i in range(len(self.heatmap_sequences)):
            sequence_length = len(self.heatmap_sequences[i])
            
            # シーケンスの長さがウィンドウサイズより大きい場合のみ処理
            if sequence_length >= window_size:
                # スライディングウィンドウを適用
                for start in range(0, sequence_length - window_size + 1, stride):
                    # ウィンドウ内のデータを切り出す
                    end = start + window_size
                    windowed_heatmaps.append(self.heatmap_sequences[i][start:end])
                    windowed_positions.append(self.positions_sequences[i][start:end])
                    windowed_num_people.append(self.num_people_sequences[i][start:end])
                    scenario_indices.append(i)
                    total_windows += 1
        
        print(f"スライディングウィンドウ方式で生成されたデータセット: {total_windows}サンプル")
        print(f"元のシナリオ数: {len(self.heatmap_sequences)}, 生成されたウィンドウ数: {total_windows}")
        
        # 新しいデータセットを作成して返す
        return SlidingWindowDataset(
            windowed_heatmaps, 
            windowed_positions, 
            windowed_num_people,
            scenario_indices,
            self.sensor_positions,
            window_size,
            stride,
            self.max_people
        )

class SlidingWindowDataset(Dataset):
    """スライディングウィンドウ方式で分割されたデータセット）"""
    def __init__(self, heatmap_sequences, positions_sequences, num_people_sequences, 
                 scenario_indices, sensor_positions, window_size, stride, max_people):
        self.heatmap_sequences = heatmap_sequences
        self.positions_sequences = positions_sequences
        self.num_people_sequences = num_people_sequences
        self.scenario_indices = scenario_indices
        self.sensor_positions = sensor_positions
        self.window_size = window_size
        self.stride = stride
        self.max_people = max_people
        
        # 統計情報を表示
        self.display_statistics()
        
    def __len__(self):
        return len(self.heatmap_sequences)
    
    def __getitem__(self, idx):
        """データセットからサンプルを取得"""
        heatmap_sequence = torch.FloatTensor(self.heatmap_sequences[idx])
        positions_sequence = self.positions_sequences[idx]
        num_people_sequence = self.num_people_sequences[idx]
        
        # 最後のフレームの座標データを取得
        last_frame_positions = positions_sequence[-1]
        
        # 固定長の座標配列を作成（最大人数分、残りは0埋め）
        max_people_fixed = MAX_PEOPLE  # 設定された最大人数を使用
        fixed_positions = np.zeros((max_people_fixed, 2))
        
        # 実際の人数分だけ座標を設定
        for i, (x, y) in enumerate(last_frame_positions):
            if i < max_people_fixed:
                fixed_positions[i] = [x, y]
        
        return {
            'heatmap_sequence': heatmap_sequence,
            'positions': torch.FloatTensor(fixed_positions),  # 固定長の座標配列
            'num_people': len(last_frame_positions),  # 実際の人数
            'scenario_idx': self.scenario_indices[idx]
        }
    
    def display_statistics(self):
        """データセットの統計情報を表示"""
        # 各ウィンドウの平均人数
        avg_people_per_window = [np.mean(num_people) for num_people in self.num_people_sequences]
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(avg_people_per_window, bins=20, alpha=0.7)
        plt.title('Distribution of Average People per Window')
        plt.xlabel('Average Number of People')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(self.scenario_indices, bins=len(set(self.scenario_indices)), alpha=0.7)
        plt.title('Windows per Scenario')
        plt.xlabel('Scenario Index')
        plt.ylabel('Number of Windows')
        
        plt.tight_layout()
        plt.show()
        
        # 基本統計量を出力
        print(f"ウィンドウ数: {len(self.heatmap_sequences)}")
        print(f"ウィンドウサイズ: {self.window_size}")
        print(f"ストライド: {self.stride}")
        print(f"元のシナリオ数: {len(set(self.scenario_indices))}")
        print(f"ウィンドウあたりの平均人数: {np.mean(avg_people_per_window):.2f}")

    def split_by_scenario(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """シナリオに基づいてデータセットを分割"""
        np.random.seed(seed)
        
        # シナリオインデックスの一覧を取得
        unique_scenarios = list(set(self.scenario_indices))
        np.random.shuffle(unique_scenarios)
        
        # シナリオの分割
        train_end = int(train_ratio * len(unique_scenarios))
        val_end = train_end + int(val_ratio * len(unique_scenarios))
        
        train_scenarios = unique_scenarios[:train_end]
        val_scenarios = unique_scenarios[train_end:val_end]
        test_scenarios = unique_scenarios[val_end:]
        
        # 各分割に対応するインデックスを収集
        train_indices = [i for i, scenario in enumerate(self.scenario_indices) if scenario in train_scenarios]
        val_indices = [i for i, scenario in enumerate(self.scenario_indices) if scenario in val_scenarios]
        test_indices = [i for i, scenario in enumerate(self.scenario_indices) if scenario in test_scenarios]
        
        print(f"シナリオの分割: 訓練 {len(train_scenarios)}シナリオ, 検証 {len(val_scenarios)}シナリオ, テスト {len(test_scenarios)}シナリオ")
        print(f"シナリオに基づく分割: 訓練 {len(train_indices)} サンプル, 検証 {len(val_indices)} サンプル, テスト {len(test_indices)} サンプル")
        
        return train_indices, val_indices, test_indices


# 損失関数（元コード完全準拠）
def custom_loss_with_threshold_schedule(pred_coords, pred_conf, true_coords, valid_counts, 
                                        prev_pred_coords=None, prev_matches=None,
                                        current_epoch=0, max_epochs=100):
    """
    多人数座標予測のためのカスタム損失関数 - 学習進行に応じた閾値導入版
    """
    # 段階的な閾値スケジュール設定
    progress_percentage = current_epoch / max_epochs * 100
    
    # 最初の50%のエポックでは閾値なし、50%-80%では10m、80%以降は5mの閾値
    if progress_percentage < 50:
        # 最初の50%は閾値なし
        threshold_active = False
        distance_threshold = float('inf')
    elif progress_percentage < 80:
        # 50-80%は中程度の閾値
        threshold_active = False
        distance_threshold = 10.0
    else:
        # 80%以降は厳しい閾値
        threshold_active = True
        distance_threshold = 5.0
    
    batch_size = pred_coords.size(0)
    max_people = pred_coords.size(1)
    device = pred_coords.device
    
    total_coord_loss = torch.tensor(0.0, device=device)
    total_conf_loss = torch.tensor(0.0, device=device)
    
    for b in range(batch_size):
        num_valid = int(valid_counts[b].item())
        
        if num_valid == 0:
            # 人がいない場合は信頼度をすべて0にする損失のみ
            total_conf_loss += torch.mean(pred_conf[b] ** 2)
            continue
        
        # 有効な真の座標を取得
        true_coords_valid = true_coords[b, :num_valid, :]
        
        # 予測座標との距離行列を計算
        distances = torch.cdist(pred_coords[b], true_coords_valid.float())
        
        # ハンガリアンアルゴリズムによる最適マッチング
        # PyTorchでは scipy を使用
        try:
            from scipy.optimize import linear_sum_assignment
            distances_np = distances.detach().cpu().numpy()
            pred_indices, true_indices = linear_sum_assignment(distances_np)
        except ImportError:
            # scipyが利用できない場合は貪欲法で近似
            pred_indices, true_indices = [], []
            remaining_preds = list(range(max_people))
            remaining_trues = list(range(num_valid))
            
            while remaining_preds and remaining_trues:
                min_dist = float('inf')
                best_pred, best_true = None, None
                
                for p in remaining_preds:
                    for t in remaining_trues:
                        if distances[p, t] < min_dist:
                            min_dist = distances[p, t]
                            best_pred, best_true = p, t
                
                if best_pred is not None:
                    pred_indices.append(best_pred)
                    true_indices.append(best_true)
                    remaining_preds.remove(best_pred)
                    remaining_trues.remove(best_true)
                else:
                    break
            
            pred_indices = np.array(pred_indices)
            true_indices = np.array(true_indices)
        
        # マッチした予測に対する座標損失を計算
        matched_distances = distances[pred_indices, true_indices]
        
        if threshold_active:
            # 閾値以下のマッチのみ考慮
            valid_matches = matched_distances <= distance_threshold
            if valid_matches.any():
                coord_loss = torch.mean(matched_distances[valid_matches])
            else:
                coord_loss = torch.tensor(distance_threshold, device=device)
        else:
            coord_loss = torch.mean(matched_distances)
        
        total_coord_loss += coord_loss
        
        # 信頼度損失の計算
        conf_targets = torch.zeros(max_people, device=device)
        
        # マッチした予測の信頼度は1に
        conf_targets[pred_indices] = 1.0
        
        # 信頼度のBCE損失
        conf_loss = nn.functional.binary_cross_entropy(pred_conf[b], conf_targets)
        total_conf_loss += conf_loss
    
    # バッチ全体で平均
    avg_coord_loss = total_coord_loss / batch_size
    avg_conf_loss = total_conf_loss / batch_size
    
    # 総合損失（座標損失により大きな重み）
    total_loss = 2.0 * avg_coord_loss + avg_conf_loss
    
    return total_loss, avg_coord_loss, avg_conf_loss

def train_on_sliding_window_dataset(dataset, batch_size=16, num_epochs=100, device=None):
    """スライディングウィンドウデータセットを用いた学習関数"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # データセットをシナリオベースで分割
    train_indices, val_indices, test_indices = dataset.split_by_scenario(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # サブセットを作成
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # データローダーを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # モデルの初期化
    model = EnhancedMultiPersonCoordinatePredictor(max_people=26, input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 訓練ループ
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        train_loss_epoch = 0
        num_batches = 0
        
        for batch in train_loader:
            heatmap_sequences = batch['heatmap_sequence'].to(device)
            true_positions = batch['positions'].to(device)
            valid_counts = torch.tensor([batch['num_people'][i] for i in range(len(batch['num_people']))]).to(device)
            
            optimizer.zero_grad()
            
            # 前向き計算
            pred_coords, pred_conf = model(heatmap_sequences)
            
            # 損失計算
            loss, coord_loss, conf_loss = custom_loss_with_threshold_schedule(
                pred_coords, pred_conf, true_positions, valid_counts,
                current_epoch=epoch, max_epochs=num_epochs
            )
            
            # 逆伝播
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            num_batches += 1
        
        train_loss_avg = train_loss_epoch / num_batches
        train_losses.append(train_loss_avg)
        
        # 検証フェーズ
        model.eval()
        val_loss_epoch = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                heatmap_sequences = batch['heatmap_sequence'].to(device)
                true_positions = batch['positions'].to(device)
                valid_counts = torch.tensor([batch['num_people'][i] for i in range(len(batch['num_people']))]).to(device)
                
                # 前向き計算
                pred_coords, pred_conf = model(heatmap_sequences)
                
                # 損失計算
                loss, coord_loss, conf_loss = custom_loss_with_threshold_schedule(
                    pred_coords, pred_conf, true_positions, valid_counts,
                    current_epoch=epoch, max_epochs=num_epochs
                )
                
                val_loss_epoch += loss.item()
                num_val_batches += 1
        
        val_loss_avg = val_loss_epoch / num_val_batches
        val_losses.append(val_loss_avg)
        
        scheduler.step()
        
        # ログ出力
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")
    
    return model, train_losses, val_losses

# ユーティリティ関数
def save_dataset(dataset, save_dir='./datasets'):
    """データセット保存関数"""
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(dataset, SlidingWindowDataset):
        save_data = {
            'sensor_positions': dataset.sensor_positions,
            'window_size': dataset.window_size,
            'stride': dataset.stride,
            'max_people': dataset.max_people,
            'heatmap_sequences': dataset.heatmap_sequences,
            'positions_sequences': dataset.positions_sequences,
            'num_people_sequences': dataset.num_people_sequences,
            'scenario_indices': dataset.scenario_indices
        }
        filename = f'slidwin_dataset_{len(dataset)}_window{dataset.window_size}_stride{dataset.stride}.pt'
    else:
        save_data = {
            'sensor_positions': dataset.sensor_positions,
            'max_people': dataset.max_people,
            'heatmap_sequences': dataset.heatmap_sequences,
            'positions_sequences': dataset.positions_sequences,
            'num_people_sequences': dataset.num_people_sequences
        }
        filename = f'dataset_{len(dataset)}_samples.pt'
    
    save_path = os.path.join(save_dir, filename)
    torch.save(save_data, save_path)
    print(f"Dataset saved to {save_path}")
    return save_path

def load_dataset(filepath):
    """データセット読み込み関数"""
    save_data = torch.load(filepath, weights_only=False)
    print(f"Dataset loaded from {filepath}")
    
    if 'window_size' in save_data:
        print("Loading Sliding Window Dataset")
        dataset = SlidingWindowDataset(
            save_data['heatmap_sequences'],
            save_data['positions_sequences'],
            save_data['num_people_sequences'],
            save_data['scenario_indices'],
            save_data['sensor_positions'],
            save_data['window_size'],
            save_data['stride'],
            save_data['max_people']
        )
    else:
        print("Loading Scenario-based Dataset")
        dataset = PedestrianTrackingDataset.__new__(PedestrianTrackingDataset)
        dataset.sensor_positions = save_data['sensor_positions']
        dataset.max_people = save_data['max_people']
        dataset.heatmap_sequences = save_data['heatmap_sequences']
        dataset.positions_sequences = save_data['positions_sequences']
        dataset.num_people_sequences = save_data['num_people_sequences']
    
    return dataset


def plot_training_curves(train_losses, val_losses):
    """訓練曲線の可視化"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_scenario_sample(dataset, sample_idx=0, timestep_interval=5):
    """シナリオサンプルの可視化（ヒートマップ含む）"""
    if sample_idx >= len(dataset):
        print(f"Sample index {sample_idx} is out of range. Dataset has {len(dataset)} samples.")
        return
    
    sample = dataset[sample_idx]
    heatmap_sequence = sample['heatmap_sequence'].numpy()
    positions_sequence = sample['positions_sequence']
    
    print(f"シナリオ {sample_idx}: 長さ {len(heatmap_sequence)} timesteps")
    
    # 指定間隔でヒートマップを表示
    timesteps_to_show = range(0, len(heatmap_sequence), timestep_interval)
    
    for t in timesteps_to_show:
        if t < len(positions_sequence):
            heatmap = heatmap_sequence[t]
            people_positions = positions_sequence[t]
            
            # センサーデータを再計算
            _, sensor_data = generate_heatmap(
                dataset.sensor_positions, people_positions, grid_size=(GRID_WIDTH, GRID_HEIGHT)
            )
            
            print(f"\nTimestep {t}: {len(people_positions)} people")
            visualize_heatmap_with_people(
                heatmap, dataset.sensor_positions, people_positions, 
                sensor_data, grid_size=(GRID_WIDTH, GRID_HEIGHT)
            )


# === メイン実行部（元コード完全準拠） ===

print("=== リファクタリング版 歩行者追跡システム ===")
# 1. シナリオベースのデータセットを作成
print("1. シナリオベースデータセット作成中...")
scenario_dataset = PedestrianTrackingDataset(
    num_samples=10,  # 実際の使用時はより多くのサンプルを推奨
    max_people=3,
    visualize=False  # Trueにすると1番目のシナリオが可視化されます
)

# 2. スライディングウィンドウデータセットに変換
print("\n2. スライディングウィンドウデータセット作成中...")
sliding_window_dataset = scenario_dataset.create_sliding_window_dataset(
    window_size=5,  # 時系列長
    stride=1         # 1ステップずつスライド（1Hz相当）
)

# 3. データセットを保存
print("\n3. データセット保存中...")
saved_path = save_dataset(sliding_window_dataset)

# 4. 保存したデータセットをロード
print("\n4. データセットロード中...")
loaded_dataset = load_dataset(saved_path)

# 5. モデルを学習
print("\n5. モデル学習開始...")
model, train_losses, val_losses = train_on_sliding_window_dataset(
    loaded_dataset,
    batch_size=8,
    num_epochs=50
)

# 6. 学習結果の可視化
print("\n6. 学習結果可視化...")
plot_training_curves(train_losses, val_losses)

# 7. モデルを保存
print("\n7. モデル保存中...")
save_path = Path('./models')
save_path.mkdir(exist_ok=True)
torch.save(model.state_dict(), save_path / 'final_model_sliding_window.pth')
print(f"Model saved to {save_path / 'final_model_sliding_window.pth'}")

# 8. サンプル可視化（オプション）
print("\n8. サンプル可視化...")
print("シナリオサンプルを表示します（最初のシナリオの一部）:")
visualize_scenario_sample(scenario_dataset, sample_idx=0, timestep_interval=10)

print("\n=== 完了 ===")

