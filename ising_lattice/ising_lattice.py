import numpy as np
import matplotlib.pyplot as plt
import fire


class IsingLattice:

    def __init__(self, x, y, J, T):
        self.x = x
        self.y = y
        self.J = J
        self.T = T
        self.dimensions = [x, y]
        self.lattice = np.random.choice(
            [-1, 1], size=self.dimensions
        )  # randomly choose a lattice with dimensions x and y
        self.unfixed_entries = []
        self.number_fixed_entries = x * y - len(
            self.unfixed_entries
        )  # number of unfixed entries

    def find_neighbours(self, spin_to_flip):
        i = spin_to_flip[0]
        j = spin_to_flip[1]
        neighbours = []
        if i in range(1, self.x - 1):
            neighbours.append([i - 1, j])
            neighbours.append([i + 1, j])
        elif i == 0:
            neighbours.append([self.x - 1, j])
            neighbours.append([i + 1, j])
        else:
            neighbours.append([i - 1, j])
            neighbours.append([0, j])

        if j in range(1, self.y - 1):
            neighbours.append([i, j - 1])
            neighbours.append([i, j + 1])
        elif j == 0:
            neighbours.append([i, self.y - 1])
            neighbours.append([i, j + 1])
        else:
            neighbours.append([i, j - 1])
            neighbours.append([i, 0])
        return neighbours

    def energy_change(self):
        for s in self.unfixed_entries:
            neighbours = self.find_neighbours(s)
            E_old = 0
            E_new = 0
            for i in neighbours:
                E_old += (
                    -1 * (self.J) * self.lattice[tuple(s)] * self.lattice[tuple(i)]
                )  # Energy of the old configuration
                E_new += (
                    -1 * (self.J) * self.lattice[tuple(s)] * self.lattice[tuple(i)] * -1
                )  # Energy of the new configuration
            delta_E = E_new - E_old
            if delta_E < 0:  # Flip the spin if the energy is lower
                self.lattice[tuple(s)] = -self.lattice[tuple(s)]
            elif delta_E > 0:
                if np.random.rand() < np.exp(
                    -delta_E / self.T
                ):  # Flip with probability e^(-deltaE/T)
                    self.lattice[tuple(s)] = -self.lattice[tuple(s)]
            else:
                self.lattice[tuple(s)] = -self.lattice[
                    tuple(s)
                ]  # flip the spin if the energy is the same

    def simulate(self, n):
        spins = np.array(
            [self.lattice[tuple(i)] for i in self.unfixed_entries]
        )  # get the spins of the unfixed entries
        for i in range(n):
            self.energy_change()
            updated_spins = np.array(
                [self.lattice[tuple(i)] for i in self.unfixed_entries]
            )  # get the spins of the unfixed entries
            spins = spins + updated_spins
        spins = (1 / n) * spins  # get the average of the spins
        for i in range(len(spins)):
            if spins[i] > 0.1:
                self.lattice[tuple(self.unfixed_entries[i])] = 1
            elif spins[i] < -0.1:
                self.lattice[tuple(self.unfixed_entries[i])] = -1
            else:
                print("The system is not in equilibrium")
        return self.lattice

    def plot_lattice(self):
        plt.imshow(self.lattice, cmap="bwr", interpolation="nearest")
        plt.colorbar(ticks=[-1, 1])
        plt.show()


if __name__ == "__main__":
    fire.Fire(IsingLattice)
