from titanic.basic.event import Event
from titanic.basic.evaluation import ModelEvaluation, evaluate
import random


def test_model(model, events: [Event]):
    for event in events:
        p = model.estimate_probability(event)
        event.probability = p
        event.score = 1 if p > 0.5 else 0


def train_model(model, events: [Event]):
    for event in events:
        model.consider(event)


def run_kfold_testing(model, events, k=5) -> [ModelEvaluation]:
    random.shuffle(events)
    total_events = len(events)
    iteration_training_size = int(((float(k)-1.0)/float(k)) * total_events)
    iteration_testing_size = total_events - iteration_training_size
    out: [ModelEvaluation] = []
    for iteration in range(0, k):
        start_index = int(iteration * iteration_testing_size)
        end_index = int(start_index + iteration_testing_size)
        if end_index > total_events:
            end_index = total_events
        current_test_events = events[start_index: end_index]
        if len(current_test_events) < 1:
            continue
        current_training_events = events[0:start_index]
        for x in events[end_index:]:
            current_training_events.append(x)
        model.reset()
        train_model(model, current_training_events)
        model = model.finish_training()
        test_model(model, current_test_events)

        evaluation = evaluate(current_test_events)
        out.append(evaluation)
    return out
