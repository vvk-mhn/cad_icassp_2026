[1.1.0] - 2025-10-02

**Summary**: This release refines metadata consistency, improves scoring text normalization, and adds track references for training.

**Changed**

- The word **WONT** (without an apostrophe) was previously accepted as valid in **v1.0.0** because it appeared in the pronunciation dictionary. Starting from **v1.1.0**, it is treated as a misspelling of **WON'T**.
- The fields **prompt** and **response** previously contained the original text provided by the annotators and listener panel. These are now stored in **original_prompt** and **original_response**.
- The fields **prompt** and **response** now contain the _normalized_ text used for scoring. 

**Added**

- The training metadata now includes the field **fma**, which corresponds to the track ID from the FMA dataset. This field is not included in the validation or evaluation sets.
  
[1.0.0] - 2025-09-01

- Initial release
