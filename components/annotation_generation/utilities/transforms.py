import numpy as np

class ChopDatBreathRightOnUp:
    '''
    Truncates x, y, & t to have sequence length == `max_length`. If the length
    is greater, takes the entire middle breath and takes (`max_length` - `middle breath length`) / 2 contiguous
    sequences from both the prior and subsequent breath. If the length is less, we left justify our window of size `max_length`
    and return that slice of the triplet.
    Args:
        justification (str): Either 'left' (window starts at the first index of the middle breath)
                             or 'center' (window centered around the midpoint of the middle breath)
        max_length (int): Maximum sequence length to truncate to
        offset (int): The number of spectral columns to shift to the left
                      of the first index of the middle breath when justification='left'
                      and to the left of the center index of the middle breath when
                      justification='center'
        jitter (int): Shift the entire window by a random number between -jitter / 2 and jitter / 2 ONLY IN TRAINING
    '''

    def __init__(self, justification='left', max_length=900, offset=300):
        # set attributes
        if justification not in ['left', 'center']:
            raise Exception('Justification must be either "left" or "center"')

        self.justification = justification
        self.max_length = max_length
        self.offset = offset

    def forward(self, xs, ys, ts, uid):

        # Get the indices of all the non-nan y values
        non_nan_ys = np.where(~np.isnan(ys))[0]

        # If we're left justifying, take the first index of the middle breath (and offset it)
        if self.justification == 'left':
            start_idx = non_nan_ys[0] - self.offset
        lenx = len(xs)
        # If we're center justifying, find the center index of the middle
        # breath, calculate the start index and offset it
        if self.justification == 'center':
            current_breath_mid_idx = non_nan_ys[(len(non_nan_ys) - 1) // 2]
            num_idx_left_of_mid = self.max_length // 2

            start_idx = current_breath_mid_idx - num_idx_left_of_mid - self.offset

        end_idx = start_idx + self.max_length

        # If our start index is out of bounds, we'll left pad the xs, ys and ts
        num_cols_to_pad_left = abs(start_idx)

        # Adjust start_idx and end_idx based on the number of
        # columns/values added to the tensor/arrays
        start_idx += num_cols_to_pad_left
        end_idx += num_cols_to_pad_left

        # If our adjusted end_idx is out of bounds we'll pad to the right
        num_cols_to_pad_right = max(0, ((end_idx - xs.shape[-1]) + 1))

        # Pad xs, ys and ts to the left and right (if padding is unnecessary the
        # values in num_cols_to_pad will be 0, resulting in no padding
        num_cols_to_pad = [num_cols_to_pad_left, num_cols_to_pad_right]

        xs = np.pad(xs, [[0, 0], [0, 0], num_cols_to_pad], mode='constant', constant_values=0)
        ys = np.pad(ys.astype(float), (num_cols_to_pad, [0, 0]), mode='constant', constant_values=np.nan)
        ts = np.pad(ts.astype(float), num_cols_to_pad, mode='constant', constant_values=np.nan)

        # Keep the slices of xs, ys and ts we care about (xs is a tensor and we want to truncate the third dimension)
        xs = xs[:, :, start_idx:end_idx]
        ys = ys[start_idx:end_idx]
        ts = ts[start_idx:end_idx]

        # Check that our window came out the right size
        assert xs.shape[-1] == len(ys) == len(
            ts) == self.max_length, 'xs: {}, ys: {}, ts: {}, original triplet: {}, non-nan ys: {}, start_idx: {}, end_idx: {}'.format(
            xs.shape[-1], len(ys), len(ts), lenx, len(non_nan_ys), start_idx, end_idx)

        return xs, ys, ts, uid