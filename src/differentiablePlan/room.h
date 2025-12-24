#include <iostream>
#include <vector>

namespace room{


    struct Room{
        size_t room_id;
        const char *name;
        std::vector<size_t> connect_room_ids;
        bool is_public;
        size_t floor_height = 1;
        
        Room(const size_t room_id_,const char* name_,const std::vector<size_t>& connect_room_ids_)
        :room_id(room_id_),name(name_),connect_room_ids(connect_room_ids_),is_public(true){};


    };

    struct Room2D : public Room{
        

    };


}