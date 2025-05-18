import random
import math
MAX_FITNESS_FOR_MUTATION = 50 
POPULATION_COUNT = 1000
MIN_WEIGHT = 0.5
Raw_Chromosome = list()

MAX_WEIGHT = float(input())
MIN_VALUE = float(input())
MIN_NUM_OF_FOOD = int(input())
MAX_NUM_OF_FOOD = int(input())

LOWER_WEIGHT_BOUND = 1
UPPER_WEIGHT_BOUND = 20
LOWER_VALUE_BOUND = 1 
UPPER_VALUE_BOUND = 20
LOWER_MIN_FOOD_BOUND = 2 
UPPER_MIN_FOOD_BOUND = 5 
NUM_OF_GENERATIONS = 1000

# MAX_WEIGHT = random.randint(LOWER_WEIGHT_BOUND,UPPER_WEIGHT_BOUND)
# MIN_VALUE = random.randint(LOWER_VALUE_BOUND,UPPER_VALUE_BOUND)
# MIN_NUM_OF_FOOD = random.randint(LOWER_MIN_FOOD_BOUND,UPPER_MIN_FOOD_BOUND)
# MAX_NUM_OF_FOOD = random.randint(MIN_NUM_OF_FOOD,10)

print("MAX_WEIGHT: ",MAX_WEIGHT)
print("MIN_VALUE: ",MIN_VALUE)
print("MIN_NUM_OF_FOOD: ", MIN_NUM_OF_FOOD)
print("MAX_NUM_OF_FOOD: ",MAX_NUM_OF_FOOD)

POPULATION = list()
NUM_OF_GENS = 0
ANSWER_FITNESS = 100
NUM_OF_DIV =  MAX_NUM_OF_FOOD - MIN_NUM_OF_FOOD + 1
MUTATION_FOR_BAD_FITNESS = 0.5
MUTATION_FOR_GOOD_FITNESS = 0.25
CROSSOVER_FOR_BAD_FITNESS = 0.8
CROSSOVER_FOR_GOOD_FITNESS = 0.75
DIFF_PENALTY = 5
MUTATION_SET = [-2,-1,1,2]
CORRECT_VALUE_FOR_MUTATION = 2
class Gen:
    def __init__(self, name:str, max_weight:float,value:float):
        self.name = name
        self.max_weight = float(max_weight)
        self.vpw = float (value) / float(max_weight)

    def get_value_per_weight(self):
        return self.vpw
    def get_name(self):
        return self.name
    def __str__(self):
        return "__________\n"+\
                f"name: {self.name}\n"+\
                f"max_weight: {self.max_weight}\n"+\
                f"value_per_weight: {self.vpw}"
    
class Chromosome:
    def __init__(self):
        self.gens_weights = [0] * NUM_OF_GENS
        self.fitness = 0 
        self.actives_count = 0
    def get_fitness_value(self):
        return self.calc_fitness()
    def count_actives(self):
        self.actives_count = 0 
        for i in range(NUM_OF_GENS):
            if(self.gens_weights[i] != 0):
                self.actives_count += 1
        return self.actives_count
    def calc_fitness(self):
        fitness = ANSWER_FITNESS
        total_weight = 0
        total_value = 0

        for i in range(len(self.gens_weights)):
            if(self.gens_weights[i] != 0):
                total_weight += self.gens_weights[i]
                total_value += (self.gens_weights[i] * Raw_Chromosome[i].get_value_per_weight() )

        weight_diff = total_weight - MAX_WEIGHT
        value_diff = MIN_VALUE - total_value
        if(weight_diff > 0 ):
            fitness -= (weight_diff * DIFF_PENALTY)
        if(value_diff > 0):
            fitness -= (value_diff * DIFF_PENALTY)
        self.fitness = fitness
        return fitness
    
    def mutation(self):
        full_temp = list()
        empty_temp = list()
        for i in range (NUM_OF_GENS):
            # print(self.gens_weights[i])
            if (self.gens_weights[i] != 0 ):
                full_temp.append(i)
            else :
                empty_temp.append(i)
        full_mut = random.randint(0,len(full_temp)-1)
        # print("len empty temp:",len(empty_temp))
        # print("empty temp:",empty_temp)
        if (len(empty_temp) == 0):
            return
            # emp_mut = 0
        else:
            emp_mut = random.randint(0,len(empty_temp)-1)
        deleted_value = self.gens_weights[full_temp[full_mut]]
        a = random.choice(MUTATION_SET) + deleted_value

        if (a > Raw_Chromosome[empty_temp[emp_mut]].max_weight ):
            self.gens_weights[empty_temp[emp_mut]] =  Raw_Chromosome[empty_temp[emp_mut]].max_weight - CORRECT_VALUE_FOR_MUTATION  
            self.gens_weights[full_temp[full_mut]] = 0

        elif(a>0):
            self.gens_weights[empty_temp[emp_mut]] = a
            self.gens_weights[full_temp[full_mut]] = 0
        self.fitness = self.get_fitness_value()
    def show_result(self):
        print(f"{'Snack':<20}{'Weight':^14}{'Value':>9}")
        total_weight = 0 
        total_value = 0 
        for i in range(NUM_OF_GENS):
            if (self.gens_weights[i] != 0 ):
                total_weight += self.gens_weights[i]
                total_value += Raw_Chromosome[i].vpw *self.gens_weights[i]
                print(f"{Raw_Chromosome[i].get_name():<20}  {self.gens_weights[i]:^10.2f} {Raw_Chromosome[i].vpw *self.gens_weights[i]:>10.2f}" )
        print(f"{'Total':<20}{total_weight:^15.2f}{total_value:>9.2f}")

    def good_for_mutation(self):
        return (self.fitness <= MAX_FITNESS_FOR_MUTATION)
    def __str__(self):
        return "__________\n"+\
                f"active_count: {self.count_actives()}\n"+\
                f"fitness: {self.fitness}\n"+\
                f"gens_weights: {self.gens_weights}"
    def __repr__(self):
        return self.__str__()


with open('snacks.csv') as file:
    content = file.readlines()
rows = content[1:]
names = [""] * len(rows)
weights = [""] * len(rows)
values = [""] * len(rows)
for i in range(len(rows)):
    names[i],weights[i],values[i] = rows[i].split(sep=",")
    values[i] = values[i].strip()
    Raw_Chromosome.append( Gen(names[i],weights[i],values[i]))
NUM_OF_GENS = len(Raw_Chromosome)


class genetic_algorithm:
    def creat_population(self):
        for i in range(NUM_OF_DIV):
            if (i != NUM_OF_DIV - 1 ):
                for k in range(POPULATION_COUNT // NUM_OF_DIV):
                    gen_num = random.sample(range(0,NUM_OF_GENS), MIN_NUM_OF_FOOD + i)
                    POPULATION.append(Chromosome ())
                    for l in range(len(gen_num)):
                        POPULATION[-1].gens_weights[gen_num[l]] = random.randint(math.ceil(MIN_WEIGHT),math.floor(Raw_Chromosome[gen_num[l]].max_weight))
            else:
                for p in range(POPULATION_COUNT - (NUM_OF_DIV -1 ) * (POPULATION_COUNT // NUM_OF_DIV)):
                    gen_num = random.sample(range(0,NUM_OF_GENS), MIN_NUM_OF_FOOD + i)
                    POPULATION.append(Chromosome ())
                    for q in range(len(gen_num)):
                        POPULATION[-1].gens_weights[gen_num[q]] = random.randint(math.ceil(MIN_WEIGHT),math.floor(Raw_Chromosome[gen_num[q]].max_weight))
            POPULATION[i].calc_fitness()

    def do_crossover(self,first_chrom:int,second_chrom:int):

        num_of_changes = POPULATION[first_chrom].actives_count / 2
        i = 0 
        while(i < num_of_changes):
            for j in range(NUM_OF_GENS):
                found = 0
                if (POPULATION[first_chrom].gens_weights[j] != 0 ):
                    for k in range(found ,NUM_OF_GENS):
                        if( j == k ):
                            if(POPULATION[first_chrom].gens_weights[k] != 0 and POPULATION[second_chrom].gens_weights[j] != 0 )or (POPULATION[first_chrom].gens_weights[k] == 0 and POPULATION[second_chrom].gens_weights[j] == 0):
                                temp = POPULATION[first_chrom].gens_weights[k]
                                POPULATION[first_chrom].gens_weights[k] = POPULATION[second_chrom].gens_weights[k]
                                POPULATION[second_chrom].gens_weights[k] = temp
                                found = k
                                i+= 1
                        else:
                            if(POPULATION[first_chrom].gens_weights[k] != 0 and POPULATION[second_chrom].gens_weights[j] != 0 )or (POPULATION[first_chrom].gens_weights[k] == 0 and POPULATION[second_chrom].gens_weights[j] == 0):
                                temp = POPULATION[first_chrom].gens_weights[k]
                                POPULATION[first_chrom].gens_weights[k] = POPULATION[second_chrom].gens_weights[k]
                                POPULATION[second_chrom].gens_weights[k] = temp

                                temp = POPULATION[first_chrom].gens_weights[j]
                                POPULATION[first_chrom].gens_weights[j] = POPULATION[second_chrom].gens_weights[j]
                                POPULATION[second_chrom].gens_weights[j] = temp
                                found = k
                                i += 2
            POPULATION[first_chrom].calc_fitness()
            POPULATION[second_chrom].calc_fitness()
            
    def mut_or_cross(self):
        changed = set()

        for i in range(POPULATION_COUNT):
            changed.add(i)
            rand_num = random.random()
            if (POPULATION[i].good_for_mutation()):
                if (rand_num <= MUTATION_FOR_BAD_FITNESS):
                    #mutation
                    POPULATION[i].mutation()

                elif(rand_num <= CROSSOVER_FOR_BAD_FITNESS):
                    div_num = math.ceil((i+ 0.1) / (POPULATION_COUNT // NUM_OF_DIV))

                    if (div_num != NUM_OF_DIV):
                        cross_couple = random.randint(i,div_num*(POPULATION_COUNT//NUM_OF_DIV))
                        start = cross_couple
                        while(cross_couple in changed):
                            if(cross_couple > div_num * (POPULATION_COUNT/NUM_OF_DIV) ):
                                cross_couple = i + 1
                            else:
                                cross_couple += 1 
                            if (cross_couple == start):
                                break     
                        self.do_crossover(i,cross_couple-1)
                        #crossover
                    else:
                        repeated = True
                        while(repeated):
                            cross_couple = random.randint(i+1,POPULATION_COUNT )
                            if (cross_couple not in changed):
                                repeated =False
                        self.do_crossover(i,cross_couple-1)
                        #crossover
                else:
                    if (rand_num <= MUTATION_FOR_GOOD_FITNESS):
                        POPULATION[i].mutation()

                        #mutation
                    elif(rand_num <= CROSSOVER_FOR_GOOD_FITNESS):
                        self.do_crossover(i,cross_couple-1)
                        #crossover
                    
    def print_result(self,answer_chrom:int):
        POPULATION[answer_chrom].show_result()
    def search_for_result(self):
        for i in range(POPULATION_COUNT):
            if(POPULATION[i].calc_fitness() == ANSWER_FITNESS):
                self.print_result(i)
                return True
            

#main:
GA = genetic_algorithm()
GA.creat_population()

for i in range(NUM_OF_GENERATIONS):
    GA.mut_or_cross()
    if(GA.search_for_result()):
        break
# print(POPULATION)