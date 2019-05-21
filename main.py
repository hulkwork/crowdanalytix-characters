import os
from src.model import simple_cnn
from src.train import train_on_gen
dir_train = 'data/CAX_Characters_Train/'
n_target = len(os.listdir(dir_train))
shape_train = (64,64,3)
model = simple_cnn(target_size=n_target, shape_target=shape_train)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_generator = train_on_gen(dir_train, shape_target=shape_train[:2])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=10
)