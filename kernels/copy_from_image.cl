kernel void copyFromImage (
   read_only image2d_t             source,
   global unsigned char * restrict destination)
{
    int2 dims = get_image_dim(source);
    size_t image_size = dims.x * dims.y;
    size_t linear_pos = get_global_id(0);
    if (linear_pos >= image_size) { return; }
    int2 coordinates = (int2) (linear_pos / dims.y, linear_pos % dims.y);
    uint4 image_element = read_imageui(source, coordinates);
    // There is only supposed to be one channel, holding unsigned chars
    destination[linear_pos] = image_element.x;
}
