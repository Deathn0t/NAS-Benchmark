from models.search_cnn import SearchCNNController
from visualize import plot

model = SearchCNNController(3, 16, 10, 20, None, n_nodes=4)
genotype = model.genotype()

print(genotype)

plot(genotype.normal, "normal_cell_genotype")
plot(genotype.reduce, "reduce_cell_genotype")
