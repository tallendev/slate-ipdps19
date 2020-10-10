#include "client_table.h"

ClientTable& ClientTable::getInstance()
{
    static ClientTable instance;
    return instance;
}

ClientTable::~ClientTable()
{
};

void ClientTable::remove(const SlateClient& it)
{
    clients.remove_if([&](SlateClient& me) { return me == it; });
}

size_t ClientTable::size()
{
    size_t s;
    s = clients.size();
    return s;
}

void ClientTable::lock_table()
{
    cl_lock.lock();
}

void ClientTable::unlock_table()
{
    cl_lock.unlock();
}

/*
template <typename T>
SlateClient ClientTable::update_clients(size_t param, T t)
{
    cl_lock.lock();
    std::for_each(clients.begin(), clients.end(), [&](const SlateClient& c) { c.update_param(param, t); });
    cl_lock.unlock();
}
*/


void ClientTable::reschedule()
{
    std::for_each(clients.begin(), clients.end(), [&](SlateClient& c) { c.reschedule(); });
}

SlateClient& ClientTable::add(int pid, CUcontext& context)
{
    clients.emplace(clients.end(), pid, context);
    SlateClient& it = clients.back();
    return it;
}

ClientTable::ClientTable() {};
