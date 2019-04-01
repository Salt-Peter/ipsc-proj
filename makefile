INCLUDE_DIR=include
CXX = g++
# -Wall = flag for all warnings
# -g = adds debugging information (so that the executable can later be run under a debugger).
CXXFLAGS = -Wall -g -std=c++1z -I$(INCLUDE_DIR)

# example:-
# all: library.cpp main.cpp
#
# $@ evaluates to all
# $< evaluates to library.cpp
# $^ evaluates to library.cpp main.cpp

#$(OBJDIR)/%.o: $(SRCDIR)/**/%.cpp
#	@echo "Compiling" $<
#	$(CXX) $(CXXFLAGS) -c $< -o $@

all:
	$(CXX) $(CXXFLAGS) src/main.cpp -o build/main.o

clean:
	rm bin/* build/*