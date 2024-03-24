import ipdb
from transformers import AutoTokenizer
# from inspect_tokenizer import min_tokens_for_string, compute_shortest_tokenizations
from packages.dp_tokenize import compute_shortest_tokenizations
from packages.tokenizer_utils import dp_tokenize_llama

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

def test_compute_shortest_tokenization_two():
    # create a test string where the derivational morpheme is at the end of the word
    input_string = "desireableish"
    V = list("desireableish") + ["desire", "able", "ish"] + ["des", "ireable", "ish"]
    shortest_tokenizations, shortest_length = compute_shortest_tokenizations(input_string, V, False, "")
    print(shortest_tokenizations)
    assert ["desire", "able", "ish"] in shortest_tokenizations
    assert ["des", "ireable", "ish"] in shortest_tokenizations

def test_llama_tokenizer():
    # TODO: watch out for multi-sentence encoding 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="hf_cache")
    input_string = "the weather"
    encoding = tokenizer.encode(input_string)
    t2i_dict = tokenizer.get_vocab()
    i2d_dict = {v: k for k, v in t2i_dict.items()}
    space_token = '▁'
    base_representation_s = list(input_string)
    # replace ' ' with a space
    # base_representation_s = [space_token if char == ' ' else char for char in base_representation_s]
    # replace with for loop
    for i, char in enumerate(base_representation_s):
        if i == 0:
            base_representation_s[i] = space_token + char
        if char == ' ':
            base_representation_s[i] = space_token

    shortest_tokenizations, length = compute_shortest_tokenizations(base_representation_s, set(t2i_dict.keys()), False, None)
    min_tokenization = shortest_tokenizations[0]
    for i in range(len(min_tokenization)):
        if i == 0:
            # strip the space token
            min_tokenization[i] = min_tokenization[i].lstrip(space_token)
        else:
            min_tokenization[i] = min_tokenization[i].replace(space_token, ' ')
    min_string = ''.join(min_tokenization)
    assert min_string == "the weather"
    assert length == 2

def test_llama_tokenize_reversal():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="hf_cache")
    i2t = {v: k for k, v in tokenizer.get_vocab().items()}
    space_token = '▁'
    t2i = tokenizer.get_vocab()
    dp_encode_llama, invert_dp_tokenize = dp_tokenize_llama(tokenizer)
    input_string = "the weather"
    encoding = dp_encode_llama(input_string)
    decoded_string = invert_dp_tokenize(encoding)
    assert decoded_string == "the weather"

    long_string = 'General thermodynamic relations for the work of polydisperse micelle formation in the model of ideal solution of molecular aggregates in nonionic surfactant solution and the model of "dressed micelles" in ionic solution have been considered.'
    default_encoding = tokenizer.encode(long_string)[1:]
    encoding = dp_encode_llama(long_string)
    decoded_string = invert_dp_tokenize(encoding)
    assert decoded_string == long_string
    assert len(encoding) <= len(default_encoding)
    print(f"Default encoding length: {len(default_encoding)}")
    print(f"DP encoding length: {len(encoding)}")
    dp_encoding = [i2t[t] for t in encoding]
    default_encoding = [i2t[t] for t in default_encoding]
    token_str = "Automated medication dispensing systems for hospital pharmacies, heralded as an important means of reducing drug errors and improving labor productivity, have also been seen as a means of furthering the transformation of the pharmacy profession from its role in dispensing prescriptions to a clinical profession concerned with treatments and patient outcomes. Automation aids this transformation by transferring the responsibility for routine dispensing to technicians performing rationalized and computer-mediated tasks. Not all pharmacists agree with these trends. Some fear a loss of professional status and employment as their knowledge is expropriated and incorporated into machinery operated by those with lesser qualifications. They fear an industrial rather than a clinical future. Their concerns are compounded by health care cutbacks. These issues were studied at two hospitals in Canada and one in France, all mid-sized public hospitals with automated unit dose drug delivery systems installed in the late 1980s and early 1990s. Preliminary results indicated national differences in approaches to hospital pharmacy automation. In Canada, pharmacists have resisted major changes in their control of the dispensing process and in their traditional roles vis à vis doctors and pharmacy technicians. In France, where hospital pharmacy as a profession is less developed than in North America, automation has brought about a far more radical substitution for pharmacists' labor."
    for token in dp_encoding:
        if token not in default_encoding:
            print(f"Token: {token} not in default encoding")

def test_llama_tokenize_reversal_2():
    input_str = "Automated medication dispensing systems for hospital pharmacies, heralded as an important means of reducing drug errors and improving labor productivity, have also been seen as a means of furthering the transformation of the pharmacy profession from its role in dispensing prescriptions to a clinical profession concerned with treatments and patient outcomes. Automation aids this transformation by transferring the responsibility for routine dispensing to technicians performing rationalized and computer-mediated tasks. Not all pharmacists agree with these trends. Some fear a loss of professional status and employment as their knowledge is expropriated and incorporated into machinery operated by those with lesser qualifications. They fear an industrial rather than a clinical future. Their concerns are compounded by health care cutbacks. These issues were studied at two hospitals in Canada and one in France, all mid-sized public hospitals with automated unit dose drug delivery systems installed in the late 1980s and early 1990s. Preliminary results indicated national differences in approaches to hospital pharmacy automation. In Canada, pharmacists have resisted major changes in their control of the dispensing process and in their traditional roles vis à vis doctors and pharmacy technicians. In France, where hospital pharmacy as a profession is less developed than in North America, automation has brought about a far more radical substitution for pharmacists' labor."
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="hf_cache")
    dp_encode_llama, invert_dp_tokenize = dp_tokenize_llama(tokenizer)
    encoding = dp_encode_llama(input_str)
    decoded_str = invert_dp_tokenize(encoding)
    assert decoded_str == input_str
    print(f"Original string: {input_str}")