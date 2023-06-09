sources: sources.o
	mpicc -o sources sources.o -lm

sources.o: sources.c 
	mpicc -c sources.c

clean:
	-rm *.o sources
