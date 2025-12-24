#include <raylib.h>
#include "util.h"

namespace renderUtil{

    Color room_color_from_id(const size_t room_id, const size_t num_room);

    std::vector<Color> room2colors(const size_t num_room);
}