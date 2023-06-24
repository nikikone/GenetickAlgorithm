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

    def fit(self) -> np.ndarray:
        flock = [np.random.randint(1, len(self._employees) + 1, size=len(self._task_lvl)) for _ in range(self._num_flock)]
        flg = [0]
        while self.check_filter(flg[0]):
            check, flg = self.fitness_check(flock)
            result = flock[flg[1]]
            flock = self.selection(check, flock)
            self._counter_iter += 1
            if self._counter_iter % 5 == 0:
                print(check[flg[1]], " - ", ' '.join(map(str, result)), flush=True)
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
        flock_return = np.array([], list)
        steck = np.array([], int)
        while len(flock_return) < len(flock):
            indexes = []
            while len(indexes) < self._num_sample_selection:
                index = np.random.randint(0, len(flock))
                if index not in indexes:
                    indexes.append(index)
            ind = check[indexes].argmax()
            if indexes[ind] not in steck:
                steck = np.append(steck, indexes[ind])
                #steck.append(indexes[ind])# = np.append(steck, flock[indexes[ind]])
                                            # steck.append(flock[indexes[ind]])
            if len(steck) >= 2:
                result = self.crossing([flock[steck[0]], flock[steck[1]]])
                mut_1 = self.mutation(result[0])
                mut_2 = self.mutation(result[1])
                if len(flock_return) == 0:
                    flock_return = np.append(flock_return, mut_1)
                    flock_return = np.vstack((flock_return, mut_2))
                else:
                    flock_return = np.vstack((flock_return, mut_1))
                    flock_return = np.vstack((flock_return, mut_2))
                steck = np.array([], int)
        return flock_return

    def crossing(self, steck) -> np.ndarray:
        #point = np.random.randint(1, len(steck[0]))
        #a_r, b_r = np.copy(steck[0]), np.copy(steck[1])
        #a_r[point:], b_r[point:] = steck[1][point:], steck[0][point:]
        #return np.array([a_r, b_r])
        first, second = 0, 0
        while first == second:
            first, second = np.random.randint(1, len(steck[0]), 2)
        if first > second:
            first, second = second, first
        a_r, b_r = np.copy(steck[0]), np.copy(steck[1])
        a_r[first:second], b_r[first:second] = steck[1][first:second], steck[0][first:second]
        return np.array([a_r, b_r])

    def mutation(self, sample) -> np.ndarray:
        samp = sample.copy()
        if np.random.randint(0, 100) < self._chance_mutation * 100:
            for iter_1, key in enumerate(samp):
                if np.random.randint(0, 100) < self._mutation_rate * 100:
                    samp[iter_1] = np.random.randint(1, len(self._employees) + 1)
        return samp
        

    def check_filter(self, res) -> bool:
        if self._counter_iter > self._max_iter: # res > self._filter_end or 
            return False
        else:
            return True