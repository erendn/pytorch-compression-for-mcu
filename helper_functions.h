#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define DEBUG 1U

void print_twoD(twoD_t *tmp, uint32_t channel)
{
#if (DEBUG)
    printf("Size: %uX%uX%u\n", tmp->channel, tmp->r, tmp->c);
    for (uint32_t i = 0; i < tmp->r; i++)
    {
        for (uint32_t j = 0; j < tmp->c; j++)
        {
            printf("%i ", (tmp->data[channel * tmp->c * tmp->r + i * tmp->c + j]));
        }
        printf("\n");
    }
#endif
}

// Choose stride>=c=r to reuse existing buffer again.
void maxPooling(twoD_t *inout, uint32_t r, uint32_t c, uint32_t stride)
{
    data_t max = 0;
    for (uint32_t i = 0; i < inout->r; i += stride)
    {
        for (uint32_t j = 0; j < inout->c; j += stride)
        {
            max = 0;
            for (uint32_t ii = 0; ii < r; ++ii)
            {
                for (uint32_t jj = 0; jj < c; ++jj)
                {
                    if (inout->data[(i + ii) * inout->c + (j + jj)] > max)
                    {
                        max = inout->data[(i + ii) * inout->c + (j + jj)];
                    }
                }
            }
            inout->data[i / stride * inout->r / stride + j / stride] = max;
        }
    }
}

void dot(twoD_t *input, const twoD_t *weights, twoD_t *output, data_t (*activation)(data_t))
{
    if (weights->r != input->r * input->c * input->channel)
    {
        printf("size mismatch\n");
    }
    output->r = weights->c;
    output->c = 1;

    data_t sum;
    uint32_t s = 0; // current sparse array index
    uint32_t sum_indices = weights->indices[0]; // sum of sparse index differences
    for (uint32_t i = 0; i < weights->c; ++i)
    {
        sum = 0;
        for (uint32_t j = 0; j < weights->r; ++j)
        {
            const uint32_t index = i * weights->r + j;
            // iterates over sparse indices until it passes the index
            while (sum_indices < index)
            {
                sum_indices += weights->indices[++s];
            }
            // if the weight at the index is nonzero, adds it to the sum
            if (sum_indices == index)
            {
                sum += (input->data[j] - input->zero_quan) * input->scale * weights->data[s];
            }
        }
        // multiplies the sum with the scale of quantized weights
        sum *= weights->scale;
        if (weights->bias)
        {
            sum += weights->bias[i];
        }
        if (activation)
        {
            sum = activation(sum);
        }
        int32_t out = sum / output->scale + output->zero_quan;
        // clips quantized activations into 8-bit bounds
        if (out > 127)
        {
            out = 127;
        }
        else if (out < -128)
        {
            out = -128;
        }
        output->data[i] = out;
    }
}

data_t reLU(data_t a)
{
    return a > 0 ? a : 0;
}

// only works when pooling stride >= r = c
void conv2D(twoD_t *input, const twoD_t *kernel, twoD_t *output, data_t (*activation)(data_t), uint32_t stride, uint32_t r, uint32_t c, uint32_t padding)
{
    uint32_t aux_r = input->r - kernel->r + 1 + 2 * padding;
    uint32_t aux_c = input->c - kernel->c + 1 + 2 * padding;
    data_t aux_sum;
    data_t max_pool;

    output->r = aux_r / stride;
    output->c = aux_c / stride;

    uint32_t s = 0; // current sparse array index
    uint32_t sum_indices = kernel->indices[0]; // sum of sparse index differences
    const uint32_t weight_size = kernel->r * kernel->c * input->channel; // size of each weight kernel
    for (uint32_t out_ch = 0; out_ch < output->channel; ++out_ch)
    {
        const uint32_t weight_begin_idx = out_ch * weight_size; // begin index of current weight kernel
        for (uint32_t i = 0; i < weight_size; ++i)
        {
            const uint32_t index = weight_begin_idx + i; // index of weights used in following loops
            // iterates over sparse indices until it passes the index
            while (sum_indices < index)
            {
                sum_indices += kernel->indices[++s];
            }
            // adds the weight value to the array
            if (sum_indices == index)
            {
                w_kernel[i] = kernel->data[s];
            }
            else
            {
                w_kernel[i] = 0;
            }
        }
        for (uint32_t i = 0; i < aux_r; i += stride)
        {
            for (uint32_t j = 0; j < aux_c; j += stride)
            {
                max_pool = 0; /* it is safe because post ReLU values are being compared */
                for (uint32_t i_pool = 0; i_pool < r; ++i_pool)
                {
                    for (uint32_t j_pool = 0; j_pool < c; ++j_pool)
                    {
                        aux_sum = 0;
                        for (uint32_t ii = 0; ii < kernel->r; ++ii)
                        {
                            for (uint32_t jj = 0; jj < kernel->c; ++jj)
                            {
                                for (uint32_t in_ch = 0; in_ch < input->channel; ++in_ch)
                                {
                                    const uint32_t index = in_ch * kernel->r * kernel->c + ii * kernel->c + jj;
                                    if (!w_kernel[index])
                                    {
                                        continue;
                                    }
                                    uint32_t sub_i = i + i_pool + ii;
                                    uint32_t sub_j = j + j_pool + jj;
                                    if ((sub_i < padding) || (sub_j < padding) || (sub_i >= (input->r + padding)) || (sub_j >= (input->c + padding)))
                                    {
                                        continue;
                                    }
                                    // multiplies quantized input activations with their scale value since integer only arithmetic overflows
                                    aux_sum += (input->data[in_ch * input->r * input->c + (sub_i - padding) * input->c + (sub_j - padding)] - input->zero_quan) * input->scale * w_kernel[index];
                                }
                            }
                        }
                        // multiplies the sum with the scale of quantized weights
                        aux_sum *= kernel->scale;
                        if (kernel->bias)
                        {
                            aux_sum += kernel->bias[out_ch];
                        }
                        if (activation)
                        {
                            aux_sum = activation(aux_sum);
                        }
                        if (aux_sum > max_pool)
                        {
                            max_pool = aux_sum;
                        }
                    }
                }
                int32_t out = max_pool / output->scale + output->zero_quan;
                // clips quantized activations into 8-bit bounds
                if (out > 127)
                {
                    out = 127;
                }
                else if (out < -128)
                {
                    out = -128;
                }
                output->data[out_ch * output->r * output->c + i / stride * output->c + j / stride] = out;
            }
        }
    }
}

uint32_t get_class(const twoD_t *output)
{
    uint32_t max_inx = 0;
    for (uint32_t i = 0; i < output->r; ++i)
    {
        if (output->data[i] > output->data[max_inx])
        {
            max_inx = i;
        }
    }
    return max_inx;
}