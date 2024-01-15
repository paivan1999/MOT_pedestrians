from math import sqrt
from time import time

import numpy as np
import random


def tt(time_start):
    return time() - time_start
def closest_binary_search(start_value, trans_cond_for_min_bound, type="bigger", time_bound=1, decreasing=False):
    def change_current_value(current_min_bound, current_max_bound):
        if not (current_min_bound and current_max_bound):
            return current_min_bound * (1 / 2 if decreasing else 2) if (
                not current_max_bound) else current_max_bound * (2 if decreasing else 1 / 2)
        else:
            return sqrt(current_min_bound * current_max_bound)

    t = time()
    current_value = start_value
    current_max_bound = False
    current_min_bound = False
    if trans_cond_for_min_bound(current_value):
        current_min_bound = current_value
    else:
        current_max_bound = current_value
    while tt(t) < time_bound:
        if trans_cond_for_min_bound(current_value):
            current_min_bound = current_value
        else:
            current_max_bound = current_value
        current_value = change_current_value(current_min_bound, current_max_bound)
    return current_max_bound if type == "bigger" else current_min_bound


class HCPColorGrid():
    def __init__(self, rgb, radius, plane_normal, row_normal):
        self.radius = radius
        self.plane_normal = plane_normal
        self.row_normal = row_normal
        self.row = 3 - plane_normal - row_normal
        self.rgb = rgb

    def get_differences(self):
        differences_local = np.array([
            [self.steps[self.row], 0.0, 0.0],
            [self.shifts[self.row], self.steps[self.row_normal], 0.0],
            [0.0, self.steps[self.row_normal] - self.shifts[self.row_normal], self.steps[self.plane_normal]],
            [self.shifts[self.row], self.shifts[self.row_normal], self.steps[self.plane_normal]]
        ])
        differences = np.zeros((4, 3))
        for diff_init, diff_result in zip(differences_local, differences):
            diff_result[[self.row, self.row_normal, self.plane_normal]] = diff_init
        return differences

    def calculate_stretch(self, find_stretched_colors=False):
        actual_size = self.rgb - self.real_remainders
        stretch = np.array([color / actual if actual else 1.0 for color, actual in zip(self.rgb, actual_size)])
        differences = self.get_differences()
        self.minimal_distance = min([np.linalg.norm(stretch * diff) for diff in differences])
        self.effect = self.minimal_distance / self.radius
        if find_stretched_colors:
            self.stretched_colors = [color * stretch for color in self.colors]

    def count_colors(self):
        usual_planes_count = (self.counts[self.plane_normal] + 1) // 2
        shifted_planes_count = self.counts[self.plane_normal] // 2
        usual_rows_in_usual_plane_count = (self.counts[self.row_normal] + 1) // 2
        shifted_rows_in_usual_plane_count = self.counts[self.row_normal] // 2
        if self.shift_is_in_remainder[self.row_normal]:
            usual_rows_in_shifted_plane_count = shifted_rows_in_usual_plane_count
            shifted_rows_in_shifted_plane_count = usual_rows_in_usual_plane_count
        else:
            usual_rows_in_shifted_plane_count = (self.counts[self.row_normal] - 1) // 2
            shifted_rows_in_shifted_plane_count = self.counts[self.row_normal] // 2
        usual_row_len = self.counts[self.row]
        shifted_row_len = self.counts[self.row] - (not self.shift_is_in_remainder[self.row])
        count_in_one_usual_plane = usual_row_len * usual_rows_in_usual_plane_count + shifted_row_len * shifted_rows_in_usual_plane_count
        count_in_one_shifted_plane = usual_row_len * usual_rows_in_shifted_plane_count + shifted_row_len * shifted_rows_in_shifted_plane_count
        self.colors_count = count_in_one_usual_plane * usual_planes_count + count_in_one_shifted_plane * shifted_planes_count

    def calculate_structure(self, find_colors=False):
        plane_normal_step = sqrt(2 / 3)
        row_normal_step = sqrt(3) / 2
        row_step = 1
        steps = np.array([0.0, 0.0, 0.0])
        steps[[self.row, self.row_normal, self.plane_normal]] = np.array([row_step, row_normal_step, plane_normal_step])
        self.steps = steps * self.radius
        plane_normal_shift = 0
        row_normal_shift = sqrt(3) / 6
        row_shift = 1 / 2
        shifts = np.array([0.0, 0.0, 0.0])
        shifts[[self.row, self.row_normal, self.plane_normal]] = np.array(
            [row_shift, row_normal_shift, plane_normal_shift])
        self.shifts = shifts * self.radius
        self.counts = np.int32(np.floor(self.rgb / self.steps) + 1)
        self.remainders = self.rgb - (self.counts - 1) * self.steps
        self.shift_is_in_remainder = self.shifts <= self.remainders
        self.real_remainders = self.remainders - self.shift_is_in_remainder * self.shifts
        self.count_colors()
        if find_colors:
            self.colors = []
            plane_normal_is_shifted = False
            for plane in range(self.counts[self.plane_normal]):
                row_normal_is_shifted = plane % 2
                for row in range(self.counts[self.row_normal] - (
                        not self.shift_is_in_remainder[self.row_normal] and row_normal_is_shifted)):
                    row_is_shifted = row % 2 != row_normal_is_shifted
                    for point in range(
                            self.counts[self.row] - (not self.shift_is_in_remainder[self.row] and row_is_shifted)):
                        is_shifted = np.array([0, 0, 0])
                        is_shifted[[self.row, self.row_normal, self.plane_normal]] = np.array(
                            [row_is_shifted, row_normal_is_shifted, plane_normal_is_shifted])
                        numb_coordinate = np.array([0, 0, 0])
                        numb_coordinate[[self.row, self.row_normal, self.plane_normal]] = np.array([point, row, plane])
                        color = self.shifts * is_shifted + self.steps * numb_coordinate
                        self.colors.append(color)
        self.calculate_stretch(find_stretched_colors=find_colors)
        # self.print_properties()

    def print_properties(self):
        print(f"radius       {self.radius}")
        print(f"row,row_normal,plane_normal     {(self.row, self.row_normal, self.plane_normal)}")
        print(f"steps --- {list(self.steps)}")
        print(f"shifts --- {list(self.shifts)}")
        print(list(self.rgb))
        print(f"remainders --- {list(self.remainders)}")
        print(f"shift_is_in_remainder --- {list(self.shift_is_in_remainder)}")
        print(f"counts --- {list(self.counts)})")
        print(f"minimal_distance --- {self.minimal_distance}")
        print(f"colors_count --- {self.colors_count}")
        print(f"effect --- {self.effect}")
        for color in self.stretched_colors:
            print(f"      {list(color)}")
        # print(f" --- {self.stretched_colors}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


class ColorsSearcher():
    def __init__(self, colors_count, rgb, **kwargs):
        self.colors_count_to_find = colors_count
        self.rgb = rgb

    def count_range_for_this_radius(self, radius):
        structures = []
        for row_normal, plane_normal in [(0, 1), (0, 2), (1, 2)]:
            structures.append(HCPColorGrid(self.rgb, radius, plane_normal, row_normal))
            structures[-1].calculate_structure(find_colors=False)
        # for structure in structures:
        #     assert structure.colors_count == len(structure.colors)
        counts = [structure.colors_count for structure in structures]
        return (min(counts), max(counts))

    def get_structures_for_this_radius(self, radius):
        structures = []
        for row_normal, plane_normal in [(0, 1), (0, 2), (1, 2)]:
            structures.append(HCPColorGrid(self.rgb, radius, plane_normal, row_normal))
            structures[-1].calculate_structure(find_colors=False)
        # for structure in structures:
        #     assert structure.colors_count == len(structure.colors)
        counts = [structure.colors_count for structure in structures]
        return [counts, structures]

    def search_radius_for_closest_bigger_min_count(self, time_bound):
        def indicator_for_min_bound(radius):
            return min(self.get_structures_for_this_radius(radius)[0]) <= self.colors_count_to_find

        return closest_binary_search(max(self.rgb) * 3, indicator_for_min_bound, type="bigger", time_bound=time_bound,
                                     decreasing=True)

    def search_radius_for_closest_smaller_max_count(self, time_bound):
        def indicator_for_min_bound(radius):
            return max(self.get_structures_for_this_radius(radius)[0]) < self.colors_count_to_find

        return closest_binary_search(max(self.rgb) * 3, indicator_for_min_bound, type="smaller", time_bound=time_bound,
                                     decreasing=True)

    def log_search_for_best_radius(self, count):
        min_bound = self.search_radius_for_closest_bigger_min_count(1)
        max_bound = self.search_radius_for_closest_smaller_max_count(1)
        log_grid = np.logspace(np.log10(min_bound), np.log10(max_bound), num=count)
        max_length = 0
        best_structure = None
        for radius in log_grid:
            for count, structure in zip(*self.get_structures_for_this_radius(radius)):
                if count >= self.colors_count_to_find:
                    if max_length < structure.minimal_distance:
                        max_length = structure.minimal_distance
                        print(max_length)
                        best_structure = structure
        return best_structure

    def transform_to_initial_rgb(self, count):
        structure = self.log_search_for_best_radius(count)
        new_structure = HCPColorGrid(structure.rgb, structure.radius, structure.plane_normal, structure.row_normal)
        new_structure.calculate_structure(find_colors=True)
        colors = new_structure.stretched_colors
        return [color / self.rgb for color in colors]


def get_k_colors(k):
    CS = ColorsSearcher(k, np.array([6 ** (1 / 4), 2, 6 ** (1 / 4)]))
    colors = CS.transform_to_initial_rgb(1000)
    sampled_colors = random.sample(colors, k)
    return map(lambda color:(round(color[0]*255),round(color[1]*255),round(color[2]*255)), sampled_colors)
    # return list(map(lambda color: f"rgb({color[0]},{color[1]},{color[2]})", to255colors))