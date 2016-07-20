CC = g++
CPPFLAGS = -Wall -g -O3 -fPIC -std=c++11 -march=native -fopenmp
INCLUDES = -I. -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux
#LDFLAGS = -L. -L/usr/lib/jvm/java-1.6.0-openjdk-1.6.0.34.x86_64/jre/lib/amd64/server/ -pthread -lz -ljvm -lhdfs -fopenmp
LDFLAGS = -L. -L${JAVA_HOME}/lib/amd64/server/ -pthread -lz -ljvm -lhdfs -fopenmp

all: ftrl_fm_train ftrl_fm_predict  
src/ftrl_train.o: src/ftrl_train.cpp src/*.h
	$(CC) -c src/ftrl_train.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

src/ftrl_predict.o: src/ftrl_predict.cpp src/*.h
	$(CC) -c src/ftrl_predict.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

src/stopwatch.o: src/stopwatch.cpp src/stopwatch.h
	$(CC) -c src/stopwatch.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

ftrl_fm_train: src/ftrl_train.o src/stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

ftrl_fm_predict: src/ftrl_predict.o src/stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)
ftrl_fm_predict_single: src/ftrl_predict_single.o src/stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)
ftrl_fm_feature: src/ftrl_fm_feature.o src/stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -f src/*.o ftrl_train ftrl_predict ftrl_fm_feature
