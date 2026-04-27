#include <iostream>
#include <set>
#include <list>
#include "../../include/tensor.h"
#include "../../include/autograd.h"

void topo_sort(Tensor *t, std::set<Tensor*>& visited, std::list<Tensor*>& topo) {

    if (!visited.count(t)) {
        if (!t || !t->prev) return;

        int num_inputs = t->prev->num_inputs;
        visited.insert(t);

        for (int i = 0; i < num_inputs; i++) {
            if (t->prev->inputs[i]->requires_grad)
                topo_sort(t->prev->inputs[i], visited, topo);
        }
        topo.push_back(t);
    }
    return;
}

void backward(Tensor *t) {
    std::set<Tensor *> visited;
    std::list<Tensor *> topo;
    topo_sort(t, visited, topo);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor *t = *it;
        if (t && t->prev && t->prev->backward) {
            t->prev->backward(t->prev, t->grad);
        }
    }

}   