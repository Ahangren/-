import chromadb
print(chromadb.config.Settings().persist_directory)


# 连接到持久化数据库
client = chromadb.PersistentClient()

# 列出所有集合并逐个删除
collections = client.list_collections()
for collection in collections:
    client.delete_collection(collection.name)
print("所有集合已删除！")