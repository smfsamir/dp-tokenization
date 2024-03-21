from inspect_tokenizer import min_tokens_for_string, compute_shortest_tokenizations

def test_quadratic_algorithm():
    vocabulary = {"a", "ab", "abc", "d", "cd", "bcd", "b", "c"}
    input_string = "abcd"
    assert min_tokens_for_string(input_string, vocabulary) == 2
    input_string_two = "adcbdab"
    assert min_tokens_for_string(input_string_two, vocabulary) == 6

    input_string = "abdcd"
    assert min_tokens_for_string(input_string, vocabulary) == 3

    vocabulary = {"un", "desirable", "und", "es", "ira", "ble", "u", "n"}
    input_string = "undesirable"
    assert min_tokens_for_string(input_string, vocabulary) == 2

    vocabulary = {"un", "desirable", "und", "es", "ira", "ble", "ish"}
    input_string = "undesirableish"
    assert min_tokens_for_string(input_string, vocabulary) == 3

def test_compute_shortest_tokenization():
    V = ["un", "desirable"] + ["und", "esirable"] + list("undesirable")
    input_string = "undesirable"
    shortest_tokenizations, shortest_length = compute_shortest_tokenizations(input_string, V, False, "") 
    print(shortest_length)
    print(f"\n{shortest_tokenizations}")
    assert ["un", "desirable"] in shortest_tokenizations
    assert ["und", "esirable"] in shortest_tokenizations