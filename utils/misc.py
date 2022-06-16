# -*- coding: utf-8 -*-

def print_part_of_a_file(filename, start_pattern, end_pattern):
    started = False
    with open(filename) as f:
        for line in f:
            if start_pattern in line:
                started = True
            if started and end_pattern in line:
                started = False
                break
            if started:
                print(line.strip())


def print_verification(filename="./OUTPUT_FILES/output_solver.txt"):
    start_pattern = "*** Verification of simulation parameters ***"
    end_pattern = "========================================="
    print_part_of_a_file(filename, start_pattern, end_pattern)
