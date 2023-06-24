import numpy as np

class GenetickAlgorithm:

    _num_flock: int
    _max_iter: int
    _filter_end: float
    _chance_mutation: float
    _mutation_rate: float
    _employees: np.ndarray # shape(n, 4)
    _task_time: np.ndarray # shape(m)
    _task_lvl: np.ndarray # shape(m)
    _counter_iter = 0
    _num_sample_selection = 4

    def __init__(self, employees, task_time, task_lvl, *, num_flock=480, max_iter=100000, filter_end=0.165, chance_mutation=0.3, mutation_rate=0.3) -> None:
        self._num_flock = num_flock
        self._max_iter = max_iter
        self._employees = employees
        self._task_time = task_time
        self._task_lvl = task_lvl
        self._filter_end = filter_end
        self._chance_mutation = chance_mutation
        self._mutation_rate = mutation_rate
        self._rng = np.random.default_rng()

    def fit(self) -> np.ndarray:
        flock = self._rng.integers(1, len(self._employees) + 1, size=(self._num_flock, len(self._task_lvl)))
        max_return, max_iter = 0, 0
        while self.check_filter(max_return):
            check, (max_return, max_iter) = self.fitness_check(flock)
            result = flock[max_iter]
            flock = self.selection(check, flock)
            self._counter_iter += 1
            if self._counter_iter % 5 == 0:
                print(check[max_iter], " - ", ' '.join(map(str, result)), flush=True)
        return result

    def fitness_check(self, group):
        max_return = 0
        iter_max = 0
        sample_return = np.zeros((len(group)))
        for iter_1, sample in enumerate(group):
            employee_time_sample = np.zeros((len(self._employees)))
            for iter_2, key in enumerate(sample):
                lvl = self._task_lvl[iter_2]
                employee_coeff = self._employees[key - 1][lvl - 1]
                employee_time_sample[key - 1] += self._task_time[iter_2] * employee_coeff # time calculating
            maxim = np.max(employee_time_sample)
            res = 100 / maxim
            if max_return < res:
                max_return = res
                iter_max = iter_1
            sample_return[iter_1] = res
        
        return sample_return, [max_return, iter_max]
    
    def selection(self, check, flock) -> np.ndarray:
        flock_return = np.zeros(shape=flock.shape, dtype=int)
        steck = np.array([], int)
        iter_flock_return = 0
        for _ in range(self._num_flock // 2):
            indexes = self._rng.choice(range(0, len(flock)), size=(2, self._num_sample_selection), replace=False)
            ind, ind_2 = check[indexes[0]].argmax(), check[indexes[1]].argmax()
            steck = np.array([indexes[0][ind], indexes[1][ind_2]], int)
            result = self.crossing([flock[steck[0]], flock[steck[1]]])
            mut_1, mut_2 = self.mutation(result[0]), self.mutation(result[1])
            flock_return[iter_flock_return] = mut_1
            flock_return[iter_flock_return + 1] = mut_2
            iter_flock_return += 2

            steck = np.array([], int)
        return flock_return

    def crossing(self, steck) -> np.ndarray:
        first, second = self._rng.choice(range(1, len(steck[0])), size=2, replace=False)
        if first > second: first, second = second, first
        a_r, b_r = np.copy(steck[0]), np.copy(steck[1])
        a_r[first:second], b_r[first:second] = steck[1][first:second], steck[0][first:second]
        return np.array([a_r, b_r])

    def mutation(self, sample) -> np.ndarray:
        samp = sample.copy()
        if self._rng.integers(0, 100) < self._chance_mutation * 100:
            mask = self._rng.integers(0, 100, len(sample)) <= self._mutation_rate * 100
            mask_2 = self._rng.integers(1, len(self._employees) + 1, len(mask))
            samp = np.where(mask, mask_2, samp)
        return samp


    def check_filter(self, res) -> bool:
        if self._counter_iter > self._max_iter: # res > self._filter_end or 
            return False
        else:
            return True