import random
import operator
import time
import matplotlib.pyplot as plt
import math
import bisect
import copy

def generatecities(length):
	cities = [[i] for i in range(0,length)]
	random.shuffle(cities)
	#switch to city 1
	a, b = cities.index([0]), cities.index(cities[0])
	cities[b], cities[a] = cities[a], cities[b]
	return cities

def generateFirstPopulation(sizePopulation, cities):
	population = []
	for i in range(sizePopulation):
		population.append(generatecities(cities))
	return population

def distance(city_coordinates, individu):
	fitness_val = 0
	for gen in range(len(individu)-1):
		fitness_val += math.sqrt(sum([(a - b) ** 2 for a, b in zip(city_coordinates[individu[gen][0]], city_coordinates[individu[gen+1][0]])]))
	fitness_val += math.sqrt(sum([(a - b) ** 2 for a, b in zip(city_coordinates[individu[len(individu)-1][0]], city_coordinates[individu[0][0]])]))
	return fitness_val

def fitnes_func(population, city_coordinates, func):
	fitness_value = [0] * len(population)
	for index_i, individu in enumerate(population):
		fitness_value[index_i] = distance(city_coordinates, individu)

	if (func== "max"):
		#minimization to maximization
		max_val = max(fitness_value)+1 #to avoid zero devision
		for index_i, individu in enumerate(population):
			fitness_value[index_i] = 1/(fitness_value[index_i])
	return [x for _,x in sorted(zip(fitness_value,population))], sorted(fitness_value)

def roullete_func(population, fitness):
	sum_fitness = sum(fitness)
	prev_prob = 0.0
	roullete = [0] * len(population)
	id_ = 1
	for idx, fitval in enumerate(fitness):
		roullete[idx] = prev_prob + (fitval/sum_fitness)
		prev_prob = roullete[idx]
		id_ += 1
	return roullete


def selection(population, fitness):
	roullete = roullete_func(population, fitness)
	parent = []
	for index_i, individu in enumerate(population):
		# r = random.uniform(roullete[0], roullete[len(roullete)-1])
		r = random.betavariate(2,1)
		selected_parent = bisect.bisect(roullete, r)
		parent.append(population[selected_parent])
	return parent

def cross_over(population, prob):
	child = []
	for index_i in range(len(population)/2):
		parent1 = population[index_i * 2]
		parent2 = population[index_i * 2 + 1]
		r = random.uniform(0.0, 1.0)
		if (r<prob):
			left = random.randint(0, len(population[0])/3)
			right = random.randint(left+1, (len(population[0])-1))
			for i in range(left, right):
				idx_parent1 = parent1.index(parent2[i])
				idx_parent2 = parent2.index(parent1[i])
				parent1[idx_parent1] = parent1[i]
				parent2[idx_parent2] = parent2[i]
				tmp = parent1[i]
				parent1[i] = parent2[i]
				parent2[i] = tmp

		child.append(parent1)
		child.append(parent2)
	# print r, "after crossover:", child
	return child

def mutation(population):
	for index_i, child in enumerate(population):
		for x in range(2):
			r = random.uniform(0.0, 1.0)
			mutated = random.randint(0, len(population[0])-1)
			tmp = child[mutated]
			if (r<0.5):
				if child[mutated][0] < (len(child)-1) & (child[mutated][0]!= 0) :
					mutationval = child[mutated][0] + 1
					idx_child = child.index([mutationval])
					child[idx_child] = tmp
					child[mutated] = [mutationval]
			else:
				# print child[mutated][0],(len(child)-1)
				if child[mutated][0] > 0 & child[mutated][0] != 0 :
					mutationval = child[mutated][0] - 1
					idx_child = child.index([mutationval])
					child[idx_child] = tmp
					child[mutated] = [mutationval]
		population[index_i] = child

	return population

def elitism(parent, child, city_coordinates, sizePopulation):
	pool = [0] * (len(parent)+len(child))
	new_population = [0] * sizePopulation
	for idx,item in enumerate(parent):
		pool[idx] = item
	for idx,item in enumerate(child):
		pool[idx+len(child)] = item
	# print pool
	sorted_population, fitness_value = fitnes_func(pool, city_coordinates, "max")
	for item in range(sizePopulation):
		new_population[item] = sorted_population[item]

	return new_population

def GA(city_coordinates, sizePopulation, evolution):
	cities = len(city_coordinates)
	population = generateFirstPopulation(sizePopulation, cities)
	sorted_population, fitness_value = fitnes_func(population, city_coordinates, "max")

	print "running GA by 1 iter best solution",sorted_population[0], "with total distance", distance(city_coordinates,sorted_population[0])

	plt.figure()
	last = 0
	for idx in range(evolution):
		# print "GA"
		sorted_population, fitness_value = fitnes_func(population, city_coordinates, "max")

		#ploted
		if (fitness_value[0] != last) | (idx == evolution-1):
			print "change in ",idx,fitness_value[0]
			point = [city_coordinates[i[0]] for i in sorted_population[0]]
			endpoint = [city_coordinates[sorted_population[0][len(sorted_population[0])-1][0]],city_coordinates[sorted_population[0][0][0]]]
			plt.plot(*zip(*point), linestyle='--', marker='o', color='b')
			plt.plot(*zip(*endpoint), linestyle='--', marker='o', color='r')

			# print fitness_value[0]
			if idx != (evolution-1):
				plt.show(block=False)
				plt.pause(0.5)
				plt.close()
			else:
				plt.show(block=True)

		last = fitness_value[0]

		parent = selection(sorted_population, fitness_value)
		child_crossover = cross_over(parent, 0.6)
		population = mutation(child_crossover)
		# population = elitism(parent, child_mutated, city_coordinates, sizePopulation)


	sorted_population, fitness_value = fitnes_func(population, city_coordinates, "max")
	print "running GA by ",evolution,"iter best solution",sorted_population[0], "with total distance", distance(city_coordinates,sorted_population[0])



sizePopulation = 50
# city_coordinates = [[15,10], [24,28], [35,35],[14,40]]
city_coordinates = [[15,10], [24,28], [35,35],[14,40],[89,35],[14,20],[65,75],[84,30],[75,35],[104,40]]
evolution = 200
GA(city_coordinates,sizePopulation, evolution)
