from qibocal.calibrations.protocols.generators import GeneratorOnequbitcliffords


def test_generators():
    """There are different generators for random circuits from a
    specific distribution. Here there are tested and shown how to use.
    """
    ############### One qubit Clifford circuits for N qubits ##################
    # 1 qubit case.
    onequbit_circuitgenerator = GeneratorOnequbitcliffords([0])
    # Use the generator (which is actually an iterator)
    # Calling the object it needs a depth/lenght of the circuit, e.g. how many
    # gates there are in the circuit.
    circuit11 = next(onequbit_circuitgenerator(1))
    circuit13 = next(onequbit_circuitgenerator(3))
    # The queue for the circuits should be one and three, respectively.
    assert len(circuit11.queue) == 1
    assert len(circuit13.queue) == 3
    # Calling the circuit generator again should give another circuit (since
    # there are only 24 Clifford gates, it may happen that they are the same).
    # Check if two randomly generated circuits are not the same, three times.
    assert not np.array_equal(
        next(onequbit_circuitgenerator(1)).unitary(),
        next(onequbit_circuitgenerator(1)).unitary(),
    ) or not np.array_equal(
        next(onequbit_circuitgenerator(1)).unitary(),
        next(onequbit_circuitgenerator(1)).unitary(),
    )
    # Check the inverse.
    onequbit_circuitgenerator_i = GeneratorOnequbitcliffords([0], invert=True)
    circuit12_i = next(onequbit_circuitgenerator_i(2))
    # The queue should be longer by one now because of the additional
    # inverse gate.
    assert len(circuit12_i.queue) == 3
    # And also the unitary of the circuit should be the identity.
    assert np.allclose(circuit12_i.unitary(), np.eye(2))
    # TODO ask Andrea what also can be tested. If the measurement is correct?
    # 5 qubits case.
    # Either give a list of qubits:
    fivequbit_circuitgenerator = GeneratorOnequbitcliffords([0, 1, 2, 3, 4])
    # Or just a number, both works.
    fivequbit_circuitgenerator1 = GeneratorOnequbitcliffords(5)
    # The qubit attribute should be the same.
    assert np.array_equal(
        fivequbit_circuitgenerator.qubits, fivequbit_circuitgenerator1.qubits
    )
    # Generate a radom circuit for 5 qubits with 1 Clifford gate each.
    circuit51 = next(fivequbit_circuitgenerator(1))
    # Generate a radom circuit for 5 qubits with 4 Clifford gate each.
    circuit54 = next(fivequbit_circuitgenerator(4))
    assert len(circuit51.queue) == 1
    assert len(circuit54.queue) == 4
    # Check the inverse.
    twoqubit_circuitgenerator_i = GeneratorOnequbitcliffords([0, 1], invert=True)
    circuit22_i = next(twoqubit_circuitgenerator_i(2))
    assert np.allclose(circuit22_i.unitary(), np.eye(2**2))
    # Try the act_on variabl.
    fourqubit_circuitgenerator = GeneratorOnequbitcliffords(
        [0, 1, 2, 3], act_on=2, invert=True
    )
    circuit41 = next(fourqubit_circuitgenerator(2))
    draw_string = circuit41.draw()
    compare_string = "q0: ─────────\nq1: ─────────\nq2: ─U─U─U─M─\nq3: ─────────"
    assert draw_string == compare_string
