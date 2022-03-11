import os
import pandas as pd

## List of gestures:

labels = {
    0: 'stand',
    1: 'jump',
    2: 'place',
    3: 'down',
    4: 'sit',
    5: 'come',
    6: 'paw',
}
## Angles and distances
distances = {
    0: 'down_close',
    1: 'down_far',
    2: 'middle_close',
}

labels_r = {value: key for key, value in labels.items()}
distances_r = {value: key for key, value in distances.items()}

dataset_path = '/media/yessense/Transcend/converted/'
stats = []

for person in os.listdir(dataset_path):
    for label in os.listdir(os.path.join(dataset_path, person)):
        # rename to human readable gesture name
        # os.rename(os.path.join(dataset_path, person, label), os.path.join(dataset_path, person, labels[int(label)]))
        for distance in os.listdir(os.path.join(dataset_path, person, label)):
            # distance_id = distance.split('_')[-1]
            # os.rename(os.path.join(dataset_path, person, label, distance),
            #           os.path.join(dataset_path, person, label, distances[int(distance_id)]))

            image_count = 0
            for image in os.listdir(os.path.join(dataset_path, person, label, distance)):
                image_count += 1

            stats.append([int(person), labels_r[label], distances_r[distance], image_count])



df = pd.DataFrame(stats, )

print("Done")

