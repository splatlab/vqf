# vqf
Vector Quotient Filters: Overcoming the Time/Space Trade-Off in Filter Design

This work appeared at SIGMOD 2021. If you use this software please cite us:
```
@inproceedings{PandeyCDB21,
  author    = {Prashant Pandey and
               Alex Conway and
               Joe Durie and
               Michael A. Bender and
               Martin Farach-Colton and
               Rob Johnson},
  title     = {Vector Quotient Filters: Overcoming the Time/Space Trade-Off in Filter Design},
  booktitle={Proceedings of the 2021 ACM international conference on Management of Data},
  year      = {2021},
}
```

Overview
--------
 The VQF supports approximate membership testing of
 items in a data set. The VQF is based on Robin Hood hashing, like the quotient
 filter, but uses power-of-two-choices hashing to reduce the variance of 
 runs, and thus offers consistent, high throughput across load factors.
 Power-of-two-choices hashing also makes it more amenable to concurrent updates.

API
--------
* 'vqf_insert(item)': insert an item to the filter
* 'vqf_is_present(item)': return the existence of the item. Note that this
  method may return false positive results like Bloom filters.
* 'vqf_remove(item)': remove the item. 

Build
-------
This library depends on libssl. 

The code uses AVX512 instructions to speed up operatons. However, there is also
an alternate implementation based on AVX2. 

```bash
 $ make main
 $ ./main 24
```

To build the code with thread-safe insertions:
```bash
 $ make THREAD=1 main_tx
 $ ./main_tx 24 4
```

 The argument to main is the log of the number of slots in the VQF. For example,
 to create a VQF with 2^30 slots, the argument will be 30.

Contributing
------------
Contributions via GitHub pull requests are welcome.


Authors
-------
- Prashant Pandey <ppandey@berkeley.edu>
- Alex Conway <aconway@vmware.com>
- Rob Johnson <robj@vmware.com>
