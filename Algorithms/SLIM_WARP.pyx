import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t
from libc.math cimport log2
from libc.stdlib cimport rand
from libc.math cimport fabs
from libc.stdio cimport printf
from libc.time cimport time_t, time as c_time

cdef inline int cython_random(int n):
    return rand() % n
cdef int[:] precompute_nonzero_indices(URM_train_coo):
    """Precompute the list of non-zero interaction indices."""
    cdef int[:] nonzero_indices = np.arange(URM_train_coo.nnz, dtype=np.int32)
    np.random.shuffle(nonzero_indices)  # Shuffle for randomness
    return nonzero_indices
cdef inline int32_t binary_search(int32_t key, int32_t[:] array, int start, int end):
    """
    Performs binary search on a sorted array segment.

    Returns:
        The index of the key if found, else -1.
    """
    cdef int mid
    while start < end:
        mid = (start + end) // 2
        if array[mid] == key:
            return mid
        elif array[mid] < key:
            start = mid + 1
        else:
            end = mid
    return -1
cdef inline int32_t sample_violation(
    int32_t user_id,
    int32_t item_id,
    double[:, :] item_item_S,
    int32_t[:] indptr,
    int32_t[:] indices,
    int n_items,
    int32_t max_attempts=100
):
    """
    Samples a violating negative item for WARP loss.

    Parameters:
        user_id (int32_t): ID of the user.
        item_id (int32_t): ID of the positive item.
        item_item_S (double[:, :]): Item-item similarity matrix.
        indptr (int32_t[:] ): CSR indptr array.
        indices (int32_t[:] ): CSR indices array.
        n_items (int): Total number of items.
        max_attempts (int32_t): Maximum attempts to find a valid negative sample.

    Returns:
        int32_t: Negative item ID if found, else -1.
    """
    cdef int attempts = 0
    cdef int32_t neg_item_id = -1
    cdef int start_idx = indptr[user_id]
    cdef int end_idx = indptr[user_id + 1]

    while attempts < max_attempts:
        neg_item_id = cython_random(n_items)
        # Perform binary search to check if neg_item_id is in user's interactions
        if binary_search(neg_item_id, indices, start_idx, end_idx) == -1:
            return neg_item_id  # Valid negative sample
        attempts += 1
    return -1  # No valid sample found after max_attempts
def do_some_training_WARP(
    URM_train,
    double initial_learning_rate,
    double regularization,
    double decay_rate,
    int num_iterations,
    double[:, :] existing_item_item_S,
    double gamma  # Margin parameter for WARP
):
    cdef int n_items = URM_train.shape[1]
    cdef int n_users = URM_train.shape[0]
    URM_train_csr = URM_train.tocsr()
    URM_train_coo = URM_train.tocoo()
    # Precompute non-zero indices
    cdef int[:] nonzero_indices = precompute_nonzero_indices(URM_train_coo)
    cdef int nnz_count = URM_train_coo.nnz
    cdef int32_t[:] indices = URM_train_csr.indices.view(dtype=np.int32)
    cdef int32_t[:] indptr = URM_train_csr.indptr.view(dtype=np.int32)
    cdef double[:] data = URM_train_csr.data.view(dtype=np.float64)
    cdef int32_t[:] coo_row = URM_train_coo.row.view(dtype=np.int32)
    cdef int32_t[:] coo_col = URM_train_coo.col.view(dtype=np.int32)

    cdef double[:, :] item_item_S
    if existing_item_item_S is not None:
        item_item_S = existing_item_item_S
    else:
        item_item_S = np.zeros((n_items, n_items), dtype=np.float64)

    cdef double learning_rate = initial_learning_rate
    cdef double loss = 0.0
    cdef int user_id, item_id, neg_item_id
    cdef int index
    cdef double predicted_rating, neg_predicted_rating, violation
    cdef int rank
    cdef double weight
    cdef int patience_counter = 0
    cdef double last_loss = float('inf')
    cdef int patience = 20
    cdef double min_delta = 1e-5
    cdef double current_loss
    cdef int start_idx
    cdef int end_id
   
    cdef long elapsed_time
    cdef int next_sample_num = 0
    cdef int sample_num = 0
    cdef time_t t
    cdef long start_time = c_time(&t)
    start_time = c_time(&t)
    cdef int samplex_index = 0
    for sample_num in range(num_iterations):
        next_sample_num = sample_num + 1

        # Sample a positive interaction from precomputed non-zero indices
        sample_index = nonzero_indices[sample_num % nnz_count]
        user_id = coo_row[sample_index]
        item_id = coo_col[sample_index]

        # Compute predicted rating for positive item
        predicted_rating = 0.0
        start_idx = indptr[user_id]
        end_idx = indptr[user_id + 1]
        for index in range(start_idx, end_idx):
            profile_item_id = indices[index]
            predicted_rating += item_item_S[profile_item_id, item_id]

        # Sample a violating negative item
        neg_item_id = sample_violation(user_id, item_id, item_item_S, indptr, indices, n_items)
        if neg_item_id == -1:
            continue  # No valid negative sample found

        # Compute predicted rating for negative item
        neg_predicted_rating = 0.0
        for index in range(start_idx, end_idx):
            profile_item_id = indices[index]
            neg_predicted_rating += item_item_S[profile_item_id, neg_item_id]

        # Check margin violation
        violation = gamma - (predicted_rating - neg_predicted_rating)
        if violation <= 0:
            continue  # No violation, skip update

        # Compute rank-based weight
        # For a more accurate rank, implement rank calculation (complex in Cython)
        # Here, we use a placeholder rank=1
        rank = 1
        weight = log2(rank + 1)

        # Update item-item similarities
        for index in range(start_idx, end_idx):
            profile_item_id = indices[index]
            # Positive item update
            item_item_S[profile_item_id, item_id] += learning_rate * (
                weight * 1.0  # Assuming binary relevance
                - regularization * item_item_S[profile_item_id, item_id]
            )
            # Negative item update
            item_item_S[profile_item_id, neg_item_id] -= learning_rate * (
                weight * 1.0
                + regularization * item_item_S[profile_item_id, neg_item_id]
            )

        # Accumulate loss
        loss += violation

        # Learning rate decay and early stopping
        if (next_sample_num) % 5000 == 0 and (next_sample_num) > 0:
            learning_rate *= decay_rate
            current_loss = loss / (next_sample_num)
            if next_sample_num % 1000000 == 0:
                elapsed_time = c_time(&t) - start_time
                printf("Iteration %d: Loss = %.4f, Time Elapsed = %.2fs\n", next_sample_num, current_loss, elapsed_time)

            # Early stopping check
            if fabs(last_loss - current_loss) < min_delta:
                patience_counter += 1
                if patience_counter >= patience:
                    printf("Early stopping at iteration %d. Loss did not improve significantly.\n", next_sample_num)
                    break
            else:
                patience_counter = 0  # Reset if loss improved
            last_loss = current_loss

    return loss, item_item_S
