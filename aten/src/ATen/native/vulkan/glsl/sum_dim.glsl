#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // x=width, y=height, z=channel, w=batch
  ivec4 input_sizes;
  // dimension to sum along
  int dim;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Returns a new tensor with values summed along dimension dim
 * Dimension dim is squeezed
 * For each pos:
 *  - Iterate over the out_texel and the summed dimension
 *  - For H,W; rearrange pos.x, pos.y
 *  - For C,H,W;
 *      When CHW are summed, batch moves into channel
 *      The src N is determined by pos.z * 4 + out_index
 */

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  int flattened_channels = int(ceil(uBlock.input_sizes.z / 4.0));
  vec4 out_texel = vec4(0, 0, 0, 0);

  // Batch
  if (uBlock.dim == 0) {
    for (int out_index = 0; out_index < 4; out_index++) {
      for (int batch = 0; batch < uBlock.input_sizes.w; batch++) {
        // src_n = batch
        // src_c = pos.z
        int src_z = batch * flattened_channels + pos.z;
        vec4 v = texelFetch(uInput, ivec3(pos.x, pos.y, src_z), 0);
        out_texel[out_index] += v[out_index];
      }
    }
    imageStore(uOutput, pos, out_texel);
  }

  // Channel
  else if (uBlock.dim == 1) {
    for (int out_index = 0; out_index < 4; out_index++) {
      for (int channel = 0; channel < uBlock.input_sizes.z; channel++) {
        // src_n = pos.z * 4 + out_index
        // src_c = channel
        int src_z =
            (pos.z * 4 + out_index) * flattened_channels + int(channel / 4);
        vec4 v = texelFetch(uInput, ivec3(pos.x, pos.y, src_z), 0);
        out_texel[out_index] += v[channel % 4];
      }
    }
    imageStore(uOutput, pos, out_texel);
  }

  // Height
  else if (uBlock.dim == 2) {
    for (int out_index = 0; out_index < 4; out_index++) {
      // src_n: pos.z * 4 + out_index
      // src_c: pos.y
      int src_z = (pos.z * 4 + out_index) * flattened_channels + pos.y / 4;
      for (int height = 0; height < uBlock.input_sizes.y; height++) {
        vec4 v = texelFetch(uInput, ivec3(pos.x, height, src_z), 0);
        out_texel[out_index] += v[pos.y % 4];
      }
    }
    imageStore(uOutput, pos, out_texel);
  }

  // Width
  else if (uBlock.dim == 3) {
    for (int out_index = 0; out_index < 4; out_index++) {
      // src_n: pos.z * 4 + out_index
      // src_c: pos.y
      int src_z = (pos.z * 4 + out_index) * flattened_channels + pos.y / 4;
      for (int width = 0; width < uBlock.input_sizes.x; width++) {
        vec4 v = texelFetch(uInput, ivec3(width, pos.x, src_z), 0);
        out_texel[out_index] += v[pos.y % 4];
      }
    }
    imageStore(uOutput, pos, out_texel);
  }
}
