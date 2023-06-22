"""
Wolf-Sheep Predation Model
================================

Replication of the model found in NetLogo:
    Wilensky, U. (1997). NetLogo Wolf Sheep Predation model.
    http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation.
    Center for Connected Learning and Computer-Based Modeling,
    Northwestern University, Evanston, IL.
"""

import mesa
import numpy as np

from AgentBased3trophic.scheduler import RandomActivationByTypeFiltered
from AgentBased3trophic.agents import Sheep, Wolf, GrassPatch


class WolfSheep(mesa.Model):
    """
    Wolf-Sheep Predation Model
    """

    height = 200
    width = 200

    initial_sheep = 10000
    initial_wolves = 3500

    sheep_reproduce = 0.04
    wolf_reproduce = 0.05

    wolf_gain_from_food = 20

    grass = True
    #grass = False
    grass_regrowth_time = 30
    sheep_gain_from_food = 4

    #verbose = True  # Print-monitoring
    verbose = False  # Print-monitoring

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        width=200,
        height=200,
        initial_sheep=10000,
        initial_wolves=3500,
        #width=20,
        #height=20,
        #initial_sheep=100,
        #initial_wolves=50,
        sheep_reproduce=0.04,
        wolf_reproduce=0.05,
        wolf_gain_from_food=20,
        grass=False,
        grass_regrowth_time=30,
        sheep_gain_from_food=4,
    ):
        """
        Create a new Wolf-Sheep model with the given parameters.

        Args:
            initial_sheep: Number of sheep to start with
            initial_wolves: Number of wolves to start with
            sheep_reproduce: Probability of each sheep reproducing each step
            wolf_reproduce: Probability of each wolf reproducing each step
            wolf_gain_from_food: Energy a wolf gains from eating a sheep
            grass: Whether to have the sheep eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                 once it is eaten
            sheep_gain_from_food: Energy sheep gain from grass, if enabled.
        """
        super().__init__()
        # Set parameters
        self.width = width
        self.height = height
        self.initial_sheep = initial_sheep
        self.initial_wolves = initial_wolves
        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.wolf_gain_from_food = wolf_gain_from_food
        self.grass = grass
        self.grass_regrowth_time = grass_regrowth_time
        self.sheep_gain_from_food = sheep_gain_from_food

        self.schedule = RandomActivationByTypeFiltered(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self.datacollector = mesa.DataCollector(
            {
                "Wolves": lambda m: m.schedule.get_type_count(Wolf),
                "Sheep": lambda m: m.schedule.get_type_count(Sheep),
                "Grass": lambda m: m.schedule.get_type_count(
                    GrassPatch, lambda x: x.fully_grown
                ),
            }
        )

        # Create sheep:
        for i in range(self.initial_sheep):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.sheep_gain_from_food)
            sheep = Sheep(self.next_id(), (x, y), self, True, energy)
            self.grid.place_agent(sheep, (x, y))
            self.schedule.add(sheep)

        # Create wolves
        for i in range(self.initial_wolves):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.wolf_gain_from_food)
            wolf = Wolf(self.next_id(), (x, y), self, True, energy)
            self.grid.place_agent(wolf, (x, y))
            self.schedule.add(wolf)

        # Create grass patches
        if self.grass:
            for agent, x, y in self.grid.coord_iter():

                fully_grown = self.random.choice([True, False])

                if fully_grown:
                    countdown = self.grass_regrowth_time
                else:
                    countdown = self.random.randrange(self.grass_regrowth_time)

                patch = GrassPatch(self.next_id(), (x, y), self, fully_grown, countdown)
                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        if self.verbose:
            print(
                [
                    self.schedule.time,
                    self.schedule.get_type_count(Wolf),
                    self.schedule.get_type_count(Sheep),
                    self.schedule.get_type_count(GrassPatch, lambda x: x.fully_grown),
                ]
            )

    def run_model(self, step_count=200):

        if self.verbose:
            print("Initial number wolves: ", self.schedule.get_type_count(Wolf))
            print("Initial number sheep: ", self.schedule.get_type_count(Sheep))
            print(
                "Initial number grass: ",
                self.schedule.get_type_count(GrassPatch, lambda x: x.fully_grown),
            )

        for i in range(step_count):
            self.step()

        if self.verbose:
            print("")
            print("Final number wolves: ", self.schedule.get_type_count(Wolf))
            print("Final number sheep: ", self.schedule.get_type_count(Sheep))
            print(
                "Final number grass: ",
                self.schedule.get_type_count(GrassPatch, lambda x: x.fully_grown),
            )

    def count(self):

        self.sheep_counts = np.zeros((self.width, self.height))
        self.wolf_counts = np.zeros((self.width, self.height))
        self.grass_counts = np.zeros((self.width, self.height))

        for cell in self.grid.coord_iter():
            cell_content, x, y = cell
            sheep = [obj for obj in cell_content if isinstance(obj, Sheep)]
            sheep_count = len(sheep)
            self.sheep_counts[x][y] = sheep_count
            wolf = [obj for obj in cell_content if isinstance(obj, Wolf)]
            wolf_count = len(wolf)
            self.wolf_counts[x][y] = wolf_count
            grass_patch = [obj for obj in cell_content if isinstance(obj, GrassPatch)][0]
            if grass_patch.fully_grown:
                grass_count = 1
                self.grass_counts[x][y] = grass_count
        
        tot_sheep = self.sheep_counts.sum()
        tot_wolves = self.wolf_counts.sum()
        tot_grass = self.grass_counts.sum()

        return tot_sheep,tot_wolves,tot_grass
