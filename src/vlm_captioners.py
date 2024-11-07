from PIL import Image
import torch
import gc
import re

# Ensure CUDA operations are synchronous for debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define the VLMCaptioner class here
class Llava_Flan_captioner:
    def __init__(self, vlm_model, processor, text_model, text_tokenizer, device='cuda'):

        self.processor = processor
        self.vlm_model = vlm_model.half().to(device) if vlm_model else None
        self.text_tokenizer = text_tokenizer
        self.text_model = text_model.half().to(device) if text_model else None
        self.device = device
        self.vision_instruction = ""
        self.text_instruction = ""
        self.main_object_replacement = ""
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

        # Add main_object to the tokenizer vocabulary
        special_tokens_dict = {
            'additional_special_tokens': [
                self.main_object_replacement,
                self.main_object_replacement.capitalize(),
                f"{self.main_object_replacement}'s",
                f"{self.main_object_replacement}'s".capitalize()
            ]
        }

        num_added_toks = self.text_tokenizer.add_special_tokens(special_tokens_dict)
        self.text_model.resize_token_embeddings(len(self.text_tokenizer))

    def extract_main_object(self, caption):
        # Use spaCy to extract the main subject
        doc = self.nlp(caption)
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ('nsubj', 'nsubjpass'):
                    subject_phrase = ' '.join([t.text for t in token.subtree])
                    return subject_phrase
        return None


    def replace_pronouns(self, caption, replacement):
        doc = self.nlp(caption)
        new_tokens = []
        main_object_introduced = False

        for sent in doc.sents:
            # Check if the main object is introduced in the sentence
            if not main_object_introduced:
                if replacement.lower() in sent.text.lower():
                    main_object_introduced = True

            # Process tokens in the sentence
            for token in sent:
                token_text = token.text
                if main_object_introduced:
                    # Replace pronouns that likely refer to the main object
                    if token.pos_ == 'PRON' and token.tag_ in {'PRP', 'PRP$'}:
                        # Handle possessive pronouns
                        if token.tag_ == 'PRP$':
                            if token.text[0].isupper():
                                token_text = replacement.capitalize() + "'s"
                            else:
                                token_text = replacement + "'s"
                        else:
                            if token.text[0].isupper():
                                token_text = replacement.capitalize()
                            else:
                                token_text = replacement
                new_tokens.append(token_text)
        # Reconstruct the caption
        new_caption = ' '.join(new_tokens)
        # Fix spacing around punctuation
        new_caption = re.sub(r'\s([?.!",\'])(?=(?:\s|$))', r'\1', new_caption)
        return new_caption.strip()


    """
    def replace_pronouns(self, caption, replacement):
        # Map pronouns to replacements
        sentences = list(self.nlp(caption).sents)
        new_caption = ""
        main_object_introduced = False
        for sent in sentences:
            sent_text = sent.text
            if not main_object_introduced:
                # Check if the sentence contains the main object
                if replacement.lower() in sent_text.lower():
                    main_object_introduced = True
            else:
                # After main object is introduced, replace pronouns
                pronoun_replacements = {
                    r'\bhe\b': replacement,
                    r'\bshe\b': replacement,
                    r'\bhim\b': replacement,
                    r'\bher\b': replacement,
                    r'\bhis\b': replacement + "'s",
                    r'\bhers\b': replacement + "'s",
                    r'\bit\b': replacement,
                    r'\bits\b': replacement + "'s",
                    r'\bthey\b': replacement,
                    r'\bthem\b': replacement,
                    r'\btheir\b': replacement + "'s",
                    r'\btheirs\b': replacement + "'s",
                }
                for pronoun_pattern, pronoun_replacement in pronoun_replacements.items():
                    sent_text = re.sub(
                        pronoun_pattern,
                        pronoun_replacement,
                        sent_text,
                        flags=re.IGNORECASE
                    )
            new_caption += sent_text + " "
        return new_caption.strip()

    """



    def replace_main_subject(self, caption, replacement):
        main_subject = self.extract_main_object(caption)
        if main_subject:
            phrases_to_replace = set()
            phrases_to_replace.add(main_subject)
            if not main_subject.lower().startswith('the '):
                phrases_to_replace.add('the ' + main_subject)
            main_subject_words = main_subject.split()
            if main_subject_words[0].lower() in ('a', 'an', 'the'):
                subject_no_article = ' '.join(main_subject_words[1:])
                phrases_to_replace.add(subject_no_article)
                phrases_to_replace.add('the ' + subject_no_article)
            # Replace main subject phrases
            for phrase in phrases_to_replace:
                pattern = r'\b' + re.escape(phrase) + r'\b'
                caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
            # Replace possessive forms
            for phrase in phrases_to_replace:
                pattern = r'\b' + re.escape(phrase) + r"'s\b"
                caption = re.sub(pattern, replacement + "'s", caption, flags=re.IGNORECASE)
            # Replace pronouns contextually
            caption = self.replace_pronouns(caption, replacement)
        else:
            print("Main subject not found in the caption.")
        return caption

    def vlm_caption(self, image_path, prompt="", resize=(768, 768), max_length=768, 
    num_beams=5, no_repeat_ngram_size=3, repetition_penalty=1.2, length_penalty = 2.0):
        try:
            # Load and resize image
            image = Image.open(image_path)
            if resize:
                image = image.resize(resize)
        except Exception as e:
            raise ValueError(f"Error opening image: {e}")

        if not prompt:
            prompt = self.vision_instruction

        with torch.no_grad(), torch.amp.autocast('cuda'):
        # torch.amp.autocast('cuda', args...)
            inputs = self.processor(prompt, 
            image, return_tensors="pt").to(self.device)

            output = self.vlm_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                length_penalty = length_penalty,
                pad_token_id =0
            )

        output_str = self.processor.decode(output[0],
                                           skip_special_tokens=True,
                                           ).strip()
        output_str = re.sub(r"\[INST\].*?\[\/INST\]", "", output_str, flags=re.DOTALL).strip()

        # Clear memory
        del inputs, output
        torch.cuda.empty_cache()
        gc.collect()

        return output_str

    def convert_caption(self, raw_caption="", instruction_prompt="", 
    max_length=512, num_beams=5, no_repeat_ngram_size=3, repetition_penalty=1.2, 
    length_penalty=2., temperature=0.9, top_p=0.9, do_sample=False,
    early_stopping=False):
        if not instruction_prompt:
            instruction_prompt = self.text_instruction

        # Replace main object if specified
        #if self.main_object_replacement:
        #    raw_caption = self.replace_main_subject(raw_caption, self.main_object_replacement)

        instruction_prompt = f"{instruction_prompt}\n{raw_caption}"
        input_ids = self.text_tokenizer(instruction_prompt, 
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=max_length,
                                        ).input_ids.to(self.device)
        
        # Ensure special token IDs are set
        if self.text_tokenizer.pad_token_id is None:
            self.text_tokenizer.pad_token_id = self.text_tokenizer.eos_token_id or 0
        if self.text_model.config.pad_token_id is None:
            self.text_model.config.pad_token_id = self.text_tokenizer.pad_token_id

        with torch.no_grad(), torch.amp.autocast('cuda'):
            output_ids = self.text_model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.text_tokenizer.pad_token_id,
                early_stopping=early_stopping

            )

        output_text = self.text_tokenizer.decode(output_ids[0],
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True  # or False, based on your preference
                                                 ).strip()
        # Replace main object if specified
        if self.main_object_replacement:
            output_text = self.replace_main_subject(output_text, self.main_object_replacement)

        # Clear memory
        del input_ids, output_ids
        torch.cuda.empty_cache()
        gc.collect()

        return output_text

    def prepare_and_convert(self, input_prompt=None, image_path=None, main_object_replacement=None,
    max_length=768, num_beams=5, no_repeat_ngram_size=3, repetition_penalty=1.2, 
    length_penalty=2.0, temperature=0.9, top_p=0.9, do_sample=False, early_stopping=False):
        if main_object_replacement:
            self.main_object_replacement = main_object_replacement

        # Generate intermediate description
        if image_path:
            try:
                intermediate_caption = self.vlm_caption(image_path=image_path, 
                prompt=self.vision_instruction)
            except ValueError as e:
                raise ValueError(f"Error generating caption from image: {e}")

        elif input_prompt:
            intermediate_caption = input_prompt
        else:
            raise ValueError("Either 'input_prompt' or 'image_path' must be provided.")

        final_caption = self.convert_caption(
            raw_caption=intermediate_caption,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            early_stopping=early_stopping
        )

        return final_caption